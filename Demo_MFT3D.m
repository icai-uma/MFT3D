% Demo of the 3D Median Filter Transform

% Coded by Karl Thurnhofer-Hemsi. June 2019.
% Variables to be considered:
%   NoiseLevel      Rician noise level present in the image
%   ZoomFactor      Grade of the upscaling of the resolution
%   ImageName       Name of the input image
%   ImageType       (Only for BrainWeb dataset) t1,t2,or pd
% Outputs:
%   	           Restored images and their stats saved in .mat files

clear all
close all
warning off

rng('default');

addpath('./competitors/');
addpath('./competitors/NLPCA');
addpath('./competitors/PRI-NLM3D');
addpath('./competitors/PRI-NLM3D/ASCM');
addpath('./competitors/PRI-NLM3D/ASCM/wavelet');
addpath('./competitors/BM4D');
addpath('./competitors/NLMUpsample');
addpath('./competitors/NLMUpsample/denoising');
addpath('./method');
addpath('./utils');
addpath('./utils/RiceOptVST');

NumMethods=7;
Methods={'odct3D';'pri_nlm3D';'mft3D';'nlm3D';'wsm';'bm4d';'nlmu'};
Labels={'odct3D';'pri-nlm3D';'mft3D';'nlm3D';'wsm';'bm4d';'nlmu'};

ImagePath = './data';


%% Model hyperparameters

NoiseLevel=5;
% RicianNoiseLevels=[1 3 5 7 9];
ZoomFactor=2;
% ZoomFactors=[2 2.5 3 3.5 4];
ImageType = 't1';
ImageName = sprintf('%s_icbm_normal_1mm_pn%g_rf0.12bits.mat',ImageType,NoiseLevel);
rician=1; % Type of noise flag

switch ZoomFactor
    case 2
        if strcmp(ImageType,'t1'), BinSize = 1.6; % T1 image
        elseif strcmp(ImageType,'t2'), BinSize = 1.6; % T2 image
        elseif strcmp(ImageType,'pd'), BinSize = 1.6; % PD image
        end 
    case 3
        if strcmp(ImageType,'t1'), BinSize = 3.15; % T1 image
        elseif strcmp(ImageType,'t2'), BinSize = 2.75; % T2 image
        elseif strcmp(ImageType,'pd'), BinSize = 2.95; % PD image
        end
    case 4
        if strcmp(ImageType,'t1'), BinSize = 4.1; % T1 image
        elseif strcmp(ImageType,'t2'), BinSize = 3.75; % T2 image
        elseif strcmp(ImageType,'pd'), BinSize = 3.95; % PD image
        end
    otherwise
        if ZoomFactor<3
            % This is for zoom factor 2.5
            if strcmp(ImageType,'t1'), BinSize = 2.55; % T1 image
            elseif strcmp(ImageType,'t2'), BinSize = 2.2; % T2 image
            elseif strcmp(ImageType,'pd'), BinSize = 2.4; % PD image
            end
        else
            if ZoomFactor<4
                % This is for zoom factor 3.5
                if strcmp(ImageType,'t1'), BinSize = 3.6; % T1 image
                elseif strcmp(ImageType,'t2'), BinSize = 3.2; % T2 image
                elseif strcmp(ImageType,'pd'), BinSize = 3.4; % PD image
                end
            end
        end
end


%% Load images

% Ground truth (only if it exists)
load(fullfile(ImagePath,sprintf('%s_icbm_normal_1mm_pn0_rf0.12bits.mat',ImageType)))
GroundTruth=double(MyData)/4095;%/255;

% Original noisy image (Brainweb)
InputImgFileName=fullfile(ImagePath,ImageName);
load(InputImgFileName)
OriginalNoisyImage=double(MyData)/double(max(max(max(MyData))));%4095;%/255;
        
% Reinicialize sizOr
sizOr = size(OriginalNoisyImage);

% Define results path and filename
ResultsPath = sprintf('./results/%s/nl%g/',...
    ImageType,NoiseLevel);
if ~exist(ResultsPath,'dir')
    mkdir(ResultsPath);
end
Results=[];
ResultsFileName=sprintf('%sResultsMFT3D_NoiseLevel%g_Zoom%g_BinSize%g.mat',...
    ResultsPath,NoiseLevel,ZoomFactor,BinSize);


%% Generation of noisy LR image

% % Add noise to GT if not enough and recalculate new noise (not for 
% Brainweb images)
% OriginalNoisyImage=ricernd(OriginalNoisyImage,0.09);
% NoiseLevel = riceVST_sigmaEst(OriginalNoisyImage)

% First step to generate LR image
Sigma = 1;
smoothedIm = imgaussfilt3(OriginalNoisyImage,Sigma);

% Second step to generate LR image
[x,y,z]=meshgrid(1:ZoomFactor:sizOr(2),1:ZoomFactor:sizOr(1),1:ZoomFactor:sizOr(3));
LowResNoisyImage = interp3(smoothedIm,x,y,z,'spline');
sizLR = size(LowResNoisyImage);

% Crop image to accomodate the dimensions to the scale factor (needed for
% NLMU method)
sizFactor = sizLR*ZoomFactor;
for i=1:3
    while (mod(sizFactor(i),1) ~= 0) || (mod(sizFactor(i),ZoomFactor) ~= 0) ...
        || (sizFactor(i)>sizOr(i))
        sizLR(i) = sizLR(i)-1;
        sizFactor = sizLR*ZoomFactor;
    end
end
LowResNoisyImage=LowResNoisyImage(1:sizLR(1),1:sizLR(2),1:sizLR(3));
sizOr=sizFactor;

GroundTruthResized=GroundTruth(1:sizOr(1),1:sizOr(2),1:sizOr(3)); % (only if it exists)
OriginalNoisyImageResized=OriginalNoisyImage(1:sizOr(1),1:sizOr(2),1:sizOr(3));


%% Start restorations
NoiseLevel = NoiseLevel/100;

for NdxMethod=1:NumMethods
    ThisMethod=Methods{NdxMethod};
    fprintf('Executing %s method...\n',ThisMethod);


    switch ThisMethod
        case 'mft3D'
            RestoredImage=MFTSuperresVarBinSize150(LowResNoisyImage,ZoomFactor,BinSize,sizOr);

        case 'nlm3D'
            v=5;
            LR_RestoredImage=MBONLM3D(LowResNoisyImage,v,1,NoiseLevel,rician);
            [x,y,z]=meshgrid(1:sizOr(2),1:sizOr(1),1:sizOr(3));

            RestoredImage=interp3(1:ZoomFactor:ZoomFactor*sizLR(2),...
                1:ZoomFactor:ZoomFactor*sizLR(1),...
                1:ZoomFactor:ZoomFactor*sizLR(3),LR_RestoredImage,x,y,z,'spline');

        case 'wsm'
            nv=3;
            fima11=MBONLM3D(LowResNoisyImage,nv,1,NoiseLevel,rician);
            fima12=MBONLM3D(LowResNoisyImage,nv,2,NoiseLevel,rician);
            LR_RestoredImage = hsm(fima11,fima12);
            [x,y,z]=meshgrid(1:sizOr(2),1:sizOr(1),1:sizOr(3));

            RestoredImage=interp3(1:ZoomFactor:ZoomFactor*sizLR(2),...
                1:ZoomFactor:ZoomFactor*sizLR(1),...
                1:ZoomFactor:ZoomFactor*sizLR(3),LR_RestoredImage,x,y,z,'spline');

        case 'odct3D'
            pack;
            LR_RestoredImage=cM_ODCT3D(LowResNoisyImage*255,NoiseLevel*255,rician);
            LR_RestoredImage=LR_RestoredImage/255;
            [x,y,z]=meshgrid(1:sizOr(2),1:sizOr(1),1:sizOr(3));

            RestoredImage=interp3(1:ZoomFactor:ZoomFactor*sizLR(2),...
                1:ZoomFactor:ZoomFactor*sizLR(1),...
                1:ZoomFactor:ZoomFactor*sizLR(3),LR_RestoredImage,x,y,z,'spline');

        case 'pri_nlm3D'
            pack;
            mv=5;
            fima=cM_ODCT3D(LowResNoisyImage*255,NoiseLevel*255,rician);
            LR_RestoredImage=cM_RI_NLM3D(LowResNoisyImage*255,mv,1,NoiseLevel*255,fima,1);
            LR_RestoredImage=LR_RestoredImage/255;
            [x,y,z]=meshgrid(1:sizOr(2),1:sizOr(1),1:sizOr(3));

            RestoredImage=interp3(1:ZoomFactor:ZoomFactor*sizLR(2),...
                1:ZoomFactor:ZoomFactor*sizLR(1),...
                1:ZoomFactor:ZoomFactor*sizLR(3),LR_RestoredImage,x,y,z,'spline');


        case 'bm4d'
            sigma = NoiseLevel; % noise standard deviation given as percentage of the maximum intensity of the signal, must be in [0,100]
            % Pero antes de llamar a la función se pone en [0,1]
            profile = 'mp'; % Default
            do_wiener = 1; % Wiener filtering
            verbose = 0; % verbose mode
            [LR_RestoredImage, sigma_est] = bm4d(LowResNoisyImage,'Rice',sigma, profile, do_wiener, verbose);
            [x,y,z]=meshgrid(1:sizOr(2),1:sizOr(1),1:sizOr(3));

            RestoredImage=interp3(1:ZoomFactor:ZoomFactor*sizLR(2),...
                1:ZoomFactor:ZoomFactor*sizLR(1),...
                1:ZoomFactor:ZoomFactor*sizLR(3),LR_RestoredImage,x,y,z,'spline');

        case 'nlmu'
            if isinteger(ZoomFactor)
                % Denoise ima
                nima1=ANLM3D(LowResNoisyImage,NoiseLevel);
                RestoredImage=NLMUpsample2(nima1,ZoomFactor*ones(1,3));
            else
                % Denoise ima
                nima1=ANLM3D(LowResNoisyImage,NoiseLevel);
                % SR to the nearest upper integer
                RestoredImage=NLMUpsample2(nima1,round(ZoomFactor)*ones(1,3));
                % Rescale
                [x,y,z]=meshgrid(1:round(ZoomFactor)/ZoomFactor:round(ZoomFactor)/ZoomFactor*sizOr(2),...
                    1:round(ZoomFactor)/ZoomFactor:round(ZoomFactor)/ZoomFactor*sizOr(1),...
                    1:round(ZoomFactor)/ZoomFactor:round(ZoomFactor)/ZoomFactor*sizOr(3));
                RestoredImage=interp3(1:round(ZoomFactor)*sizLR(2),...
                    1:round(ZoomFactor)*sizLR(1),...
                    1:round(ZoomFactor)*sizLR(3),RestoredImage,x,y,z,'spline');
            end
    end

    % Compute statistics
    Quality=RestorationQuality(255*GroundTruthResized,255*RestoredImage);
    Results.(ThisMethod).BrainMSE10=Quality.BrainMSE10;
    Results.(ThisMethod).BrainPSNR10=Quality.BrainPSNR10;
    Results.(ThisMethod).SSIM3d10=Quality.SSIM3d10;
    Results.(ThisMethod).BrainBC10=Quality.BrainBC10;

    fprintf('Results for NoiseLevel %g, ZoomFactor = %g, BinSize = %g:\n',...
        NoiseLevel,ZoomFactor,BinSize);
    fprintf('BrainMSE>10: %d\n',Quality.BrainMSE10);
    fprintf('SSIM3d>10: %d\n',Quality.SSIM3d10);
    fprintf('BC>10: %d\n',Quality.BrainBC10);

    % Conversion to the save' space
    HRGroundTruth=uint8(GroundTruthResized*255);
    HROriginalNoisyImage=uint8(OriginalNoisyImageResized*255);
    LRNoisyImage=uint8(LowResNoisyImage*255);
    RestoredImage_uint8=uint8(RestoredImage*255);

    Results.(ThisMethod).RestoredImage=RestoredImage_uint8;
    save(ResultsFileName,'Results','HRGroundTruth','HROriginalNoisyImage','LRNoisyImage','NoiseLevel','ZoomFactor','BinSize');

end

%% Graphical comparisons

for NdxMethod=1:NumMethods
    figure
    ThisLabel=Labels{NdxMethod};
    ThisMethod=Methods{NdxMethod};

    subplot(2,3,1),imshow(imrotate(GroundTruthResized(:,:,round(sizOr(3)/2)),90)),xlabel('Ground truth')
    subplot(2,3,2),imshow(imrotate(OriginalNoisyImageResized(:,:,round(sizOr(3)/2)),90)),xlabel('Original noisy image')
    subplot(2,3,3),imshow(imrotate(LowResNoisyImage(:,:,round(sizLR(3)/2)),90)),xlabel('LR noisy image')
    subplot(2,3,4),imshow(imrotate(double(Results.(ThisMethod).RestoredImage(:,:,round(sizOr(3)/2)))/255,90)),xlabel(ThisLabel)
    subplot(2,3,5),imshow(abs(imrotate(OriginalNoisyImageResized(:,:,round(sizOr(3)/2))-GroundTruthResized(:,:,round(sizOr(3)/2)),90))),xlabel('Ideal residual')
    subplot(2,3,6),imshow(abs(imrotate(OriginalNoisyImageResized(:,:,round(sizOr(3)/2))-double(Results.(ThisMethod).RestoredImage(:,:,round(sizOr(3)/2)))/255,90))),xlabel('Error residual')

    saveas(gcf,[ResultsPath,'Results_',ThisLabel,'_NoiseLevel',num2str(NoiseLevel*100),...
        '_Zoom',num2str(ZoomFactor),'_BinSize',num2str(BinSize),'.pdf'])
end

