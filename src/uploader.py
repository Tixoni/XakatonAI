import os
from typing import Optional

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    print("[UPLOAD] boto3 не установлен. Установите: pip install boto3")


def upload_file(local_path: str, remote_name: Optional[str] = None) -> bool:

    # Получаем переменные окружения
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_endpoint_url = os.getenv("AWS_ENDPOINT_URL")
    bucket_name = os.getenv("BUCKET_NAME")
    
    # Проверяем, что boto3 установлен
    if not BOTO3_AVAILABLE:
        print("[UPLOAD] Пропущено: boto3 не установлен")
        return False
    
    # Проверяем, что все переменные установлены
    if not all([aws_access_key_id, aws_secret_access_key, aws_endpoint_url, bucket_name]):
        print("[UPLOAD] Пропущено: не установлены переменные окружения для S3")
        print("[UPLOAD] Нужны: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL, BUCKET_NAME")
        return False
    
    if remote_name is None:
        remote_name = os.path.basename(local_path)
    
    try:
        session = boto3.session.Session()
        
        s3 = session.client(
            service_name='s3',
            endpoint_url=aws_endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        s3.upload_file(local_path, bucket_name, remote_name)
        print(f"[UPLOAD] {local_path} → s3://{bucket_name}/{remote_name}")
        return True
    except Exception as e:
        print(f"[UPLOAD ERROR] {e}")
        return False

