@echo off
echo ====================================
echo   Chatbot Kesehatan Ibu Hamil
echo ====================================
echo.

echo Checking system requirements...
python test_system.py

echo.
echo Press any key to start the chatbot application...
pause > nul

echo.
echo Starting Streamlit application...
echo Open your browser to: http://localhost:8501
echo.
echo Login credentials:
echo NIK: 3276011234567890
echo Password: 123456
echo.

streamlit run chatbot_app.py
