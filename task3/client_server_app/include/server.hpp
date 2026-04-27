#pragma once

#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/concurrent_hash_map.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>


template <typename T>
class Server {
public:
    using Task = std::function<T()>;

    Server() = default;

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    ~Server() {
        if (running_.load()) {
            stop();
        }
    }

    void start() {
        bool expected = false;

        if (!running_.compare_exchange_strong(expected, true)) {
            return;
        }

        reset_timers();

        worker_ = std::thread(&Server::worker_loop, this);
    }

    void stop() {
        bool expected = true;

        if (!running_.compare_exchange_strong(expected, false)) {
            return;
        }

        task_queue_.abort();

        if (worker_.joinable()) {
            worker_.join();
        }
    }

    std::size_t add_task(Task task) {
        if (!running_.load()) {
            throw std::runtime_error("Server is not running");
        }

        const std::size_t id = next_id_.fetch_add(1);

        auto state = std::make_shared<ResultState>();

        {
            typename ResultMap::accessor acc;
            results_.insert(acc, id);
            acc->second = state;
        }

        TaskItem item;
        item.id = id;
        item.task = std::move(task);

        task_queue_.push(std::move(item));

        return id;
    }

    T request_result(std::size_t id) {
        std::shared_ptr<ResultState> state;

        {
            typename ResultMap::const_accessor acc;

            if (!results_.find(acc, id)) {
                throw std::runtime_error("Unknown task id: " + std::to_string(id));
            }

            state = acc->second;
        }

        std::unique_lock<std::mutex> lock(state->mtx);

        state->cv.wait(lock, [&state] {
            return state->ready;
        });

        if (state->exception) {
            std::rethrow_exception(state->exception);
        }

        T result = *(state->value);

        lock.unlock();

        results_.erase(id);

        return result;
    }

    double server_lifetime_seconds() const {
        std::lock_guard<std::mutex> lock(stats_mtx_);
        return server_lifetime_time_.count();
    }

    double total_task_time_seconds() const {
        std::lock_guard<std::mutex> lock(stats_mtx_);
        return total_task_time_.count();
    }

    std::size_t completed_tasks() const {
        std::lock_guard<std::mutex> lock(stats_mtx_);
        return completed_tasks_;
    }

private:
    struct TaskItem {
        std::size_t id;
        Task task;
    };

    struct ResultState {
        std::mutex mtx;
        std::condition_variable cv;

        bool ready = false;
        std::optional<T> value;
        std::exception_ptr exception = nullptr;
    };

    using ResultPtr = std::shared_ptr<ResultState>;
    using ResultMap = oneapi::tbb::concurrent_hash_map<std::size_t, ResultPtr>;

private:
    void reset_timers() {
        std::lock_guard<std::mutex> lock(stats_mtx_);

        server_lifetime_time_ = std::chrono::duration<double>{0.0};
        total_task_time_ = std::chrono::duration<double>{0.0};
        completed_tasks_ = 0;
    }

    void worker_loop() {
        const auto server_start = std::chrono::steady_clock::now();

        while (running_.load()) {
            TaskItem item;

            try {
                task_queue_.pop(item);
            } catch (...) {
                break;
            }

            std::shared_ptr<ResultState> state;

            {
                typename ResultMap::const_accessor acc;

                if (!results_.find(acc, item.id)) {
                    continue;
                }

                state = acc->second;
            }

            try {
                const auto task_start = std::chrono::steady_clock::now();

                T result = item.task();

                const auto task_end = std::chrono::steady_clock::now();
                const std::chrono::duration<double> task_elapsed = task_end - task_start;

                {
                    std::lock_guard<std::mutex> stats_lock(stats_mtx_);
                    total_task_time_ += task_elapsed;
                    ++completed_tasks_;
                }

                {
                    std::lock_guard<std::mutex> lock(state->mtx);
                    state->value = result;
                    state->ready = true;
                }

                state->cv.notify_one();
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(state->mtx);
                    state->exception = std::current_exception();
                    state->ready = true;
                }

                state->cv.notify_one();
            }
        }

        const auto server_end = std::chrono::steady_clock::now();

        {
            std::lock_guard<std::mutex> lock(stats_mtx_);
            server_lifetime_time_ = server_end - server_start;
        }
    }

private:
    oneapi::tbb::concurrent_bounded_queue<TaskItem> task_queue_;
    ResultMap results_;

    std::atomic<std::size_t> next_id_{0};
    std::atomic<bool> running_{false};

    std::thread worker_;

    mutable std::mutex stats_mtx_;
    std::chrono::duration<double> server_lifetime_time_{0.0};
    std::chrono::duration<double> total_task_time_{0.0};
    std::size_t completed_tasks_ = 0;
};
