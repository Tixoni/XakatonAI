-- Создание таблицы для работников
CREATE TABLE IF NOT EXISTS workers (
    id SERIAL PRIMARY KEY,
    track_id INTEGER UNIQUE NOT NULL,
    color VARCHAR(255),
    stand_frames INTEGER DEFAULT 0,
    walk_frames INTEGER DEFAULT 0,
    work_frames INTEGER DEFAULT 0,
    attributes JSONB
);

-- Создание таблицы для поездов
CREATE TABLE IF NOT EXISTS trains (
    id SERIAL PRIMARY KEY,
    track_id INTEGER UNIQUE NOT NULL,
    number VARCHAR(50),
    total_time INTEGER DEFAULT 0
);

-- Создание индексов для улучшения производительности
CREATE INDEX IF NOT EXISTS idx_workers_track_id ON workers(track_id);
CREATE INDEX IF NOT EXISTS idx_trains_track_id ON trains(track_id);

