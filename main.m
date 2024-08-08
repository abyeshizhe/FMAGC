clear 
close all
clear all

clc

load WebKB

numView = length(X);
for i=1:numView
    X{i}=double(X{i});
    X{i}=mapstd(X{i});
end
Y=double(Y);
%% Initialization
maxIter = 3;
numView = length(X);
c = length(unique(Y));                      % number of cluster
[n,~] = size(X{1});                         % number of samples 
klist=[1,2,3,4,5,6,7,8];
gammalist=[0.9,1.1,1.3,1.5,1.7,1.9,2.1];
k=klist(3); %锚点数为2^k;
gamma = gammalist(5);                              % 1<gamma<2
alpha = ones(numView,1)/numView;
OBJJJ=[];RESULT=[];T=[];Btemp=zeros(n,n);

tic
for II=1:10
II;
OBJJ=[];
%% Get Anchor Graph
for v = 1:size(X,2)
    
    [~,d] = size(X{v}); 
    [label, Anchors] = hKM(X{v}', [1:n], k, 1);       
    B{v} = ConstructA_NP(X{v}', Anchors);
    BB{v} = B{v}*B{v}';
    Btemp = Btemp+BB{v};
end
[y0, C_centers] = litekmeans(Btemp./numView, c);
F_save = y0;   % 加载外部点时注释掉
      
%% Optimization

for Iter = 1:maxIter

% Update F
AA = zeros(n,n);
for v = 1:size(X,2)
    A{v} = alpha(v)*BB{v};
    AA = AA+A{v};
end
%[~,labInt]=max(F,[],2);
[y0, minO, iter_num, obj] = CDKM(sqrt(AA), y0,c);
F = sparse(1:n,y0,1,n,c,n);

% Update \alpha

for v = 1:numView
    W{v} = trace(B{v}'*BB{v}*B{v})-2*trace(F'*BB{v}*F*inv(F'*F));
    r = 1/(1-gamma);
    Wtemp(v) = (gamma*W{v})^r; 
end
alpha = Wtemp./(sum(Wtemp,2));

% objective value
OBJ = 0;
for v = 1:numView
    Obj(v) = sum((alpha(v)^gamma)*W{v});
    OBJ = OBJ+Obj(v);
end
OBJJ=[OBJJ, OBJ];
end
Result = ClusteringMeasure(Y, y0);
if all(diff(OBJJ)<=0)
OBJJJ=[OBJJJ,OBJJ];
RESULT = [RESULT;Result];
end

end
t=toc/10
T = [T t];
record=[mean(RESULT(:,1)),std(RESULT(:,1));
 mean(RESULT(:,2)),std(RESULT(:,2));
 mean(RESULT(:,3)),std(RESULT(:,3));
 mean(RESULT(:,4)),std(RESULT(:,4));
 mean(RESULT(:,5)),std(RESULT(:,5));
 mean(RESULT(:,6)),std(RESULT(:,6));
 mean(RESULT(:,7)),std(RESULT(:,7));
 mean(T),std(T);
 ];
record=record'
para=[k,gamma,maxIter]


