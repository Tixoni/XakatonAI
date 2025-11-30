FROM nginx:alpine

# Копируем статические файлы
COPY . /usr/share/nginx/html

# Копируем конфигурацию nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Открываем порт
EXPOSE 80

# Nginx запускается автоматически

