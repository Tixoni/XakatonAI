// Переключение вкладок
const tabButtons = document.querySelectorAll('.tabs__button');
const tabContents = document.querySelectorAll('.tab-content');

tabButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const targetTab = button.getAttribute('data-tab');

    // Убираем активный класс у всех кнопок
    tabButtons.forEach((btn) => {
      btn.classList.remove('tabs__button_active');
    });

    // Добавляем активный класс к выбранной кнопке
    button.classList.add('tabs__button_active');

    // Скрываем все контенты вкладок
    tabContents.forEach((content) => {
      content.classList.remove('tab-content_active');
    });

    // Показываем выбранный контент
    const targetContent = document.getElementById(`${targetTab}-content`);
    if (targetContent) {
      targetContent.classList.add('tab-content_active');
    }
  });
});

// Обработка выбора предприятия
const enterpriseSelect = document.getElementById('enterprise-select');
if (enterpriseSelect) {
  enterpriseSelect.addEventListener('change', (e) => {
    console.log('Выбрано предприятие:', e.target.value);
    // Здесь можно добавить логику загрузки данных для выбранного предприятия
  });
}

// Обработка кликов по видео файлам (заглушка)
const videoItems = document.querySelectorAll('.sidebar__video-placeholder');
const videoPlayer = document.querySelector('.video-player');

videoItems.forEach((item, index) => {
  item.addEventListener('click', () => {
    console.log('Выбрано видео:', index + 1);
    // Здесь можно добавить логику загрузки видео
    // Например: videoPlayer.src = 'path/to/video.mp4';
  });
});
