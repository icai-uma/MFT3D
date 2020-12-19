function [SuperResImage]=MFTSuperresVarBinSize150(InputImage,ZoomFactor,BinSize,OutputSize)
% Obtain a superresolution version of an image by the Median Filter
% Transform
% Inputs:
%   InputImage=RGB image in double format, values in the range [0,1]
%   ZoomFactor=The superresolution factor. Must be an integer larger than 1.
%   BinSize=The size of the bin. It is the length of each side of the cube which defines the bin,
%           measured in pixels in the low resolution image 
%   OutputSize=The size of the output image (only used for fractional zoom
%   factors)
% Note: set the lowermargin, uppermargin and nummedianfilter parameters
% by hand according to the chosen zoomfactor.

NumMedianFilters=150;
        
% Initializate SR image
if nargin==3
    SuperResImage=zeros(size(InputImage,1)*ZoomFactor,size(InputImage,2)*ZoomFactor,size(InputImage,3)*ZoomFactor);
else
    SuperResImage=zeros(OutputSize(1),OutputSize(2),OutputSize(3));
end

% Create grids
[X,Y,Z]=ndgrid(1:ZoomFactor:ZoomFactor*size(InputImage,1),...
    1:ZoomFactor:ZoomFactor*size(InputImage,2),...
    1:ZoomFactor:ZoomFactor*size(InputImage,3));
TrainSamples=zeros(3,numel(X));
TrainSamples(1,:)=X(:);
TrainSamples(2,:)=Y(:);
TrainSamples(3,:)=Z(:);
[X,Y,Z]=ndgrid(1:size(SuperResImage,1),1:size(SuperResImage,2),1:size(SuperResImage,3));
TestSamples=zeros(3,numel(X));
TestSamples(1,:)=X(:);
TestSamples(2,:)=Y(:);
TestSamples(3,:)=Z(:);


% Apply the Median Filter Transform
TrainFuncValues=InputImage(:);
Model=MedianFilterTransform(TrainSamples,TrainFuncValues,NumMedianFilters,BinSize);
SuperResImage=reshape(TestMFTMEXmid(Model,TestSamples),[size(SuperResImage,1) size(SuperResImage,2) size(SuperResImage,3)]);

    


