"use strict";

// Элементы интерфейса
const appealText = document.getElementById("appealText");
const characterCount = document.getElementById("characterCount");
const analyzeButton = document.getElementById("analyzeButton");
const buttonText = document.getElementById("buttonText");
const errorMessage = document.getElementById("errorMessage");

const emptyResult = document.getElementById("emptyResult");
const resultContent = document.getElementById("resultContent");
const sentimentIcon = document.getElementById("sentimentIcon");
const sentimentLabel = document.getElementById("sentimentLabel");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceProgress = document.getElementById("confidenceProgress");
const resultDescription = document.getElementById("resultDescription");

// Максимальная длина текста
const MAX_LENGTH = 10000;

// Примеры обращений
const examples = {
    positive: "Спасибо за прекрасную работу",
    neutral: "Прошу сообщить время приема",
    negative: "Возмущена качеством обслуживания"
};

// Описание возможных результатов
const sentiments = {
    POSITIVE: {
        label: "Положительная",
        icon: "☺",
        description:
            "Модель определила преимущественно положительную эмоциональную окраску текста."
    },
    NEUTRAL: {
        label: "Нейтральная",
        icon: "○",
        description:
            "Модель определила преимущественно нейтральную эмоциональную окраску текста."
    },
    NEGATIVE: {
        label: "Отрицательная",
        icon: "☹",
        description:
            "Модель определила преимущественно отрицательную эмоциональную окраску текста."
    }
};

// Обновление счётчика символов
function updateCharacterCount() {
    const length = appealText.value.length;
    characterCount.textContent =
        `${length.toLocaleString("ru-RU")} / 10 000`;
}

// Отображение ошибки
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.hidden = false;
}

// Скрытие ошибки
function hideError() {
    errorMessage.textContent = "";
    errorMessage.hidden = true;
}

// Состояние загрузки
function setLoading(isLoading) {
    analyzeButton.disabled = isLoading;

    buttonText.textContent = isLoading
        ? "Выполняется анализ..."
        : "Анализировать обращение";
}

// Отображение результата
function displayResult(data) {
    const sentimentKey = String(data.sentiment).toUpperCase();
    const sentiment = sentiments[sentimentKey];

    if (!sentiment) {
        throw new Error("Модель вернула неизвестную тональность.");
    }

    const confidence = Number(data.confidence);

    if (!Number.isFinite(confidence) ||
        confidence < 0 ||
        confidence > 1) {
        throw new Error("Модель вернула некорректную оценку уверенности.");
    }

    const percentage = confidence * 100;

    emptyResult.hidden = true;
    resultContent.hidden = false;

    resultContent.classList.remove(
        "positive",
        "neutral",
        "negative"
    );

    resultContent.classList.add(sentimentKey.toLowerCase());

    sentimentIcon.textContent = sentiment.icon;
    sentimentLabel.textContent = sentiment.label;
    resultDescription.textContent = sentiment.description;

    confidenceValue.textContent =
        `${percentage.toFixed(2).replace(".", ",")}%`;

    confidenceBar.style.width = `${percentage}%`;

    confidenceProgress.setAttribute(
        "aria-valuenow",
        String(Math.round(percentage))
    );
}

// Основная функция анализа
async function analyzeText() {
    hideError();

    const text = appealText.value.trim();

    if (!text) {
        showError("Введите текст обращения для анализа.");
        appealText.focus();
        return;
    }

    if (text.length > MAX_LENGTH) {
        showError("Текст не должен превышать 10 000 символов.");
        return;
    }

    // Скрываем предыдущий результат
    resultContent.hidden = true;
    emptyResult.hidden = false;

    setLoading(true);

    try {
        // Отправляем текст в Spring Boot
        const response = await fetch("/api/sentiment", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            if (response.status === 400) {
                throw new Error(
                    "Проверьте введённый текст обращения."
                );
            }

            throw new Error(
                `Ошибка сервера (${response.status}).`
            );
        }

        // Получаем JSON с результатом модели
        const data = await response.json();

        displayResult(data);

    } catch (error) {
        console.error("Ошибка анализа:", error);

        showError(
            error instanceof TypeError
                ? "Не удалось связаться с сервером. Проверьте, запущен ли CiviMind."
                : error.message || "Не удалось выполнить анализ."
        );

    } finally {
        setLoading(false);
    }
}

// Изменение текста
appealText.addEventListener("input", () => {
    updateCharacterCount();
    hideError();
});

// Нажатие на кнопку анализа
analyzeButton.addEventListener("click", analyzeText);

// Кнопки с примерами
document.querySelectorAll(".example-button").forEach(button => {
    button.addEventListener("click", () => {
        const exampleType = button.dataset.example;
        const exampleText = examples[exampleType];

        if (!exampleText) {
            return;
        }

        appealText.value = exampleText;
        updateCharacterCount();
        hideError();

        resultContent.hidden = true;
        emptyResult.hidden = false;

        appealText.focus();
    });
});

// Начальное состояние
updateCharacterCount();
hideError();
