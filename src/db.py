
import os
import psycopg2
from psycopg2 import OperationalError, IntegrityError
from psycopg2.extras import RealDictCursor, Json
from typing import Optional, Dict, Any, List


def get_conn():
    try:
        return psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB", "xakaton"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "12345"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432"))
        )
    except OperationalError as e:
        raise Exception(f"Ошибка подключения к базе данных: {e}")


def insert_worker(
    track_id: int,
    color: Optional[str] = None,
    stand_frames: int = 0,
    walk_frames: int = 0,
    work_frames: int = 0,
    attributes: Optional[Dict[str, List[str]]] = None
):
    
    conn = None
    cur = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        
        # Преобразуем attributes в JSONB
        attributes_json = Json(attributes) if attributes else None
        
        # Проверяем, существует ли запись
        cur.execute("SELECT id FROM workers WHERE track_id = %s", (track_id,))
        exists = cur.fetchone()
        
        if exists:
            # Обновляем существующую запись
            cur.execute("""
                UPDATE workers 
                SET color = %s, 
                    stand_frames = %s, 
                    walk_frames = %s, 
                    work_frames = %s,
                    attributes = %s
                WHERE track_id = %s
            """, (color, stand_frames, walk_frames, work_frames, attributes_json, track_id))
        else:
            # Вставляем новую запись
            cur.execute("""
                INSERT INTO workers (track_id, color, stand_frames, walk_frames, work_frames, attributes)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (track_id, color, stand_frames, walk_frames, work_frames, attributes_json))
        
        conn.commit()
    except IntegrityError as e:
        if conn:
            conn.rollback()
        raise Exception(f"Ошибка при сохранении работника: {e}")
    except Exception as e:
        if conn:
            conn.rollback()
        raise Exception(f"Ошибка при вставке работника: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def insert_train(
    track_id: int,
    number: Optional[str] = None,
    total_time: int = 0
):
    conn = None
    cur = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        
        # Проверяем, существует ли запись
        cur.execute("SELECT id FROM trains WHERE track_id = %s", (track_id,))
        exists = cur.fetchone()
        
        if exists:
            # Обновляем существующую запись
            cur.execute("""
                UPDATE trains 
                SET number = %s, 
                    total_time = %s
                WHERE track_id = %s
            """, (number, total_time, track_id))
        else:
            # Вставляем новую запись
            cur.execute("""
                INSERT INTO trains (track_id, number, total_time)
                VALUES (%s, %s, %s)
            """, (track_id, number, total_time))
        
        conn.commit()
    except IntegrityError as e:
        if conn:
            conn.rollback()
        raise Exception(f"Ошибка при сохранении поезда: {e}")
    except Exception as e:
        if conn:
            conn.rollback()
        raise Exception(f"Ошибка при вставке поезда: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

