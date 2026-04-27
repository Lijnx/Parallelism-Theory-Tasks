#include <iostream>
#include <queue>
#include <future>
#include <thread>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>
#include <condition_variable>

// Очередь задач
std::queue<std::packaged_task<int()>> tasks;
std::mutex mut; // Мьютекс для синхронизации доступа к очереди
std::condition_variable cond_var; // Условная переменная для уведомления потока сервера

// Функция вычисления степени, имитирует длительную задачу
int f(int x, int y)
{
    std::this_thread::sleep_for(std::chrono::seconds(2)); // Имитация долгой операции
    return std::pow(x, y);
}

// Поток сервера, который обрабатывает задачи из очереди
void server_thread(std::stop_token stoken)
{
    std::packaged_task<int()> task;
    std::unique_lock<std::mutex> lock(mut, std::defer_lock);

    while (!stoken.stop_requested())
    {
        lock.lock();
        cond_var.wait(lock, [&stoken] { return !tasks.empty() || stoken.stop_requested(); }); // Ожидание задачи
        
        if (stoken.stop_requested())
        {
            break;
        }

        // Проверяем, что очередь не пуста
        if (!tasks.empty())
        {
            task = std::move(tasks.front()); // Перемещаем задачу из очереди
            tasks.pop(); // Удаляем задачу из очереди
            lock.unlock(); // Разблокируем мьютекс перед выполнением задачи
            task(); // Выполняем задачу
        }
    }
    std::cout << "Server stop!\n";
}

// Поток, который добавляет задачи в очередь
void add_task_thread()
{
    // Формируем задачу
    std::packaged_task<int()> task(std::bind(f, 2, 4));
    std::future<int> result = task.get_future(); // Получаем future для получения результата
    
    {
        std::lock_guard<std::mutex> lock(mut);
        tasks.push(std::move(task)); // Добавляем задачу в очередь
    }
    cond_var.notify_one(); // Уведомляем поток сервера о новой задаче
    
    std::cout << "task_thread:\t" << result.get() << '\n'; // Ожидаем и выводим результат
}

int main()
{
    std::cout << "Start\n";

    std::jthread server(server_thread); // Запуск потока сервера
    std::thread add_task(add_task_thread); // Запуск потока добавления задач

    add_task.join(); // Ожидание завершения потока добавления задач
    server.request_stop(); // Остановка сервера
    cond_var.notify_all(); // Пробуждаем серверный поток, чтобы он мог завершиться
    server.join(); // Ожидание завершения сервера
    
    std::cout << "End\n";
}
