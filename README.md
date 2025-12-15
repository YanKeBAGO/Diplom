# Diplom

Установка

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

Запуск

Windows:

python.exe main.py result.json --outdir out --device cuda

Linux/MacOS:

python main.py result.json --outdir out --device cuda

Выход

out/report.json - итоговый профиль

out/messages_scored.csv - сообщения с оценками ML

out/report.pdf - отчёт
