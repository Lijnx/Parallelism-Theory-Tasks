# Task 6 — Heat2D

Небольшой проект с решением 2D heat diffusion / relaxation задачи.  
Один и тот же бинарник `heat2d` умеет работать в двух режимах:

- `serial` — обычный CPU single core;
- `openacc` — OpenACC-вариант для GPU или CPU multicore.

## Зависимости

Нужны:

- CMake 3.16+
- C++17 compiler
- Boost.Program_options
- NVIDIA HPC SDK, если собираете OpenACC-вариант

## Сборка

Перейдите в папку `task6` и собирайте в отдельные каталоги.

### GPU

```bash
cmake -S . -B build-gpu -DENABLE_OPENACC=ON -DOPENACC_TARGET=gpu
cmake --build build-gpu -j
```

### CPU multicore

```bash
cmake -S . -B build-multicore -DENABLE_OPENACC=ON -DOPENACC_TARGET=multicore
cmake --build build-multicore -j
```

### CPU single core

```bash
cmake -S . -B build-serial -DENABLE_OPENACC=OFF
cmake --build build-serial -j
```

## Запуск

### GPU

```bash
./build-gpu/heat2d --mode openacc --device gpu --size 1024
```

### CPU multicore

```bash
./build-multicore/heat2d --mode openacc --device multicore --size 1024
```

### CPU single core

```bash
./build-serial/heat2d --mode serial --size 1024
```

## Полезные параметры

```bash
./heat2d --help
```

Основные флаги:

- `--size` / `-n` — размер сетки `NxN`
- `--eps` / `-e` — порог сходимости
- `--iters` / `-i` — максимальное число итераций
- `--check-every` — как часто проверять ошибку
- `--save` — сохранить итоговую матрицу в файл
- `--bench` — прогон бенчмарка и запись CSV

## Примечание

Для OpenACC-сборки целевой режим задаётся через `OPENACC_TARGET`:
`gpu`, `multicore` или `host`. Для CPU single core используется режим `serial`.
