@echo off
echo Installing GPU-compatible PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo Installing other requirements...
pip install -r requirements.txt
echo Installation complete!
pause
