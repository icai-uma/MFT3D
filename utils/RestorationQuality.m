function [Results]=RestorationQuality(OrigImg,RestoredImg)
% Computes restoration quality measures for images in the [0,255] range, as in:
% C.S. Anand, J.S. Sahambi / Magnetic Resonance Imaging 28 (2010) 842–861

% Measuring performance only in brain tissue (intensity>10)
ind10 = OrigImg>10;
BrainPlainErrors10=(OrigImg(ind10)-RestoredImg(ind10));

% Mean Squared Error (MSE)
Results.BrainMSE10=mean(BrainPlainErrors10(:).^2);
Results.BrainPSNR10=10*log10((255^2)/Results.BrainMSE10);

% Structural Similarity Index (SSIM)
Results.SSIM3d10=ssim_index3d(OrigImg,RestoredImg,[1 1 1],ind10);

% Bhattacharrya coefficient (BC)
HistOrig=histc(OrigImg(ind10),0:256);
HistOrig=HistOrig/sum(HistOrig);
HistRestored=histc(RestoredImg(ind10),0:256);
HistRestored=HistRestored/sum(HistRestored);
Results.BrainBC10=sum(sqrt(HistOrig.*HistRestored));


