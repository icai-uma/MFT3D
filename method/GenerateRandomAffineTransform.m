function [A, b, Volume]=GenerateRandomAffineTransform(Dimension,BinSize)
% Generate a random affine transform. Single parameter version.
% Coded by Ezequiel Lopez-Rubio. June 2016.
% Inputs:
%   Dimension       Dimension of the input space
%   BinSize         Bin size parameter. It is the length of the side of the
%                   cube which defines the bin, measured in pixels in the
%                   original image (the low resolution image for
%                   superresolution applications).

% Generate a random translation uniformly
b=rand(Dimension,1);
% b=zeros(Dimension,1); % This makes the performance much worse

% Generate a random rotation uniformly
[Q,R]=qr(randn(Dimension));
Q=Q*diag(sign(diag(R)));
if det(Q)<0
    Q(:,1)=-Q(:,1);
end
%Q=eye(3); % This makes the performance very unstable

% Generate the scaling matrix according to the chosen bin size
Lambda=(1/BinSize)*eye(Dimension);

% Compute the transformation matrix and the bin volume
A=Q*Lambda;
Volume=1/prod(diag(Lambda));





