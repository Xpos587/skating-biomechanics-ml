// These should be caught by no-i18n-jsx-text

// JSX children with hardcoded text
<h1>Welcome</h1>
<h2>Анализ видео</h2>
<Button>Сохранить</Button>
<span>Loading...</span>
<p>Введите email для входа</p>

// String attribute with text
<Button title="Submit" />
<div aria-label="Профиль пользователя" />
