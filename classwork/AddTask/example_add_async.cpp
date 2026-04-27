#include <iostream>
#include <queue>
#include <future>
#include <thread>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>

// Мьютекс для синхронизации доступа к общим данным
std::mutex mut;

// Очередь задач: хранит пары (id задачи, асинхронный результат)
std::queue<std::pair<size_t, std::future<int>>> tasks;

// Словарь результатов вычислений
std::unordered_map<int, int> results;

// Функция вычисления степени, имитирует длительную задачу
int f(int x, int y)
{
    std::this_thread::sleep_for(std::chrono::seconds(2)); // Задержка в 2 секунды
    return std::pow(x, y);
}

// Поток сервера, который обрабатывает задачи из очереди
void server_thread(std::stop_token stoken)
{
    std::unique_lock lock_res{mut, std::defer_lock}; // Лок для защиты ресурсов
    size_t id_task;
    
    while (!stoken.stop_requested()) // Пока не поступил сигнал на остановку
    {
        lock_res.lock();
        if (!tasks.empty()) // Если в очереди есть задачи
        {
            id_task = tasks.back().first; // Получаем ID последней задачи
            results.insert({id_task, tasks.back().second.get()}); // Выполняем задачу и сохраняем результат
            tasks.pop(); // Удаляем задачу из очереди
        }
        lock_res.unlock();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Ожидание для разгрузки CPU
    }
    
    std::cout << "Server stop!\n";
}

// Поток, который добавляет задачи в очередь
void add_task_thread()
{
    size_t id_task = 1; // Идентификатор первой задачи
    std::unique_lock lock_res{mut, std::defer_lock}; // Лок для работы с ресурсами

    // Создаем задачу, которая будет выполняться лениво (deferred)
    std::future<int> result = std::async(std::launch::deferred,
                                         []() -> int
                                         {
                                             return f(2, 4);
                                         });

    // Добавляем задачу в очередь
    lock_res.lock();
    ++id_task; // Увеличиваем ID задачи
    tasks.push({id_task, std::move(result)}); // Перемещаем future в очередь
    lock_res.unlock();

    // Ожидание результата выполнения задачи
    bool ready_result = false;
    while (!ready_result)
    {
        lock_res.lock();
        if (results.find(id_task) != results.end()) // Проверяем, есть ли результат задачи
        {
            std::cout << "task_thread result:\t" << results[id_task] << '\n'; // Выводим результат
            results.erase(id_task); // Удаляем обработанный результат
            ready_result = true;
        }
        lock_res.unlock();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Ожидание для разгрузки CPU
    }
}

int main()
{
    std::cout << "Start\n";

    std::jthread server(server_thread); // Запуск сервера в отдельном потоке
    std::thread add_task(add_task_thread); // Запуск потока добавления задач

    add_task.join(); // Ожидание завершения потока добавления задач
    server.request_stop(); // Запрос остановки сервера
    server.join(); // Ожидание завершения сервера
    
    std::cout << "End\n";
}
