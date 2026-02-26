@echo off
echo Starting FUTA M.Tech Watermarking Pipeline...
echo ------------------------------------------
echo Step 1: Preprocessing Images...
python step2_preprocess.py
echo.
echo Step 2: Training Deep Learning Model (70/20/10 Split)...
python step4_train.py
echo.
echo Step 3: Generating Final Metrics (PSNR/SSIM/NC)...
python step5_results.py
echo ------------------------------------------
echo Pipeline Complete! Results are ready for Chapter 4.
pause