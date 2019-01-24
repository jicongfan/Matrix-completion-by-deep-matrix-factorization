% This is a toy example of DMF
clc
clear all
% Generate synthetic data
missrate=0.5;% missing rate
m=20;
n=100;
r=2;
x=unifrnd(-1,1,[r,n]);
X=randn(m,r)*x+(randn(m,r)*x.^2+randn(m,r)*x.^3);% polynomial function
% mask
N=size(X,2);
[nr,nc]=size(X);
M=ones(nr,nc);
for i=1:N
    temp=randperm(nr,ceil(nr*missrate));% 1
    M(temp,i)=0;
end
X0=X;% complete data (original)
X=X.*M;% incomplete data masked by M (binary matrix)
% DMF setup
s=[r 10 m];% input size, hidden size 1, ..., output size
options.Wp=0.01;
options.Zp=0.01;
options.maxiter=1000;
options.activation_func={'tanh_opt','linear'};
[X_DMF,NN_MF]=MC_DMF(X',M',s,options);
Xr=X_DMF';
% compute recovery error
re_error=norm((X0-Xr).*(1-M),'fro')/norm(X0.*(1-M),'fro');
disp(['Relative recovery error is ' num2str(re_error)])
