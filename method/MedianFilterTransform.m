function [Model]=MedianFilterTransform(Samples,FuncValues,NumMedianFilters,BinSize)
% Discrete Median Filter Transform. Version with only one parameter.
% Coded by Ezequiel Lopez-Rubio. June 2016.
% Inputs:
%   Samples         DxN matrix with N training samples of dimension D
%   FuncValues      1xN matrix with N function values
%   NumFilters      Number of filters parameter H
%   BinSize         Bin size parameter. It is the length of the side of the
%                   cube which defines the bin, measured in pixels in the
%                   original image (the low resolution image for
%                   superresolution applications).
% Outputs:
%   Model           Resulting MFT model

% Get input sizes
[Dimension,NumSamples]=size(Samples);

% Initialize model
Model.BinSize=BinSize;
Model.Dimension=Dimension;
Model.NumSamples=NumSamples;
Model.NumMedianFilters=NumMedianFilters;
Model.A=zeros(Dimension,Dimension,NumMedianFilters);
Model.b=zeros(Dimension,NumMedianFilters);
Model.VolumeBin=zeros(1,NumMedianFilters);

% Compute the median filters
for NdxHistogram=1:NumMedianFilters
    [A,b,Volume]=GenerateRandomAffineTransform(Dimension,BinSize);
    Model.MedianFilter{NdxHistogram}=ComputeMediansMEXmid(Samples,FuncValues,A,b);    
    Model.A(:,:,NdxHistogram)=A;
    Model.b(:,NdxHistogram)=b;
    Model.VolumeBin(NdxHistogram)=Volume;
end
