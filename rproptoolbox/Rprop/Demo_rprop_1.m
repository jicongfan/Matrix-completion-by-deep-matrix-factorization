% Compare Rprop to fminunc
%
%   Copyright (c) 2012 Roberto Calandra
%   $Revision: 0.55 $


%% Init

funcgrad = @onehump;

minfunc = 10e-6;
niter = 10;

p.verbosity = 0;                    % Increase verbosity to print something
p.MaxIter   = 10000;            	% Maximum number of iterations
p.d_Obj     = minfunc;
p.method    = 'IRprop-';            % Use IRprop- algorithm
p.display   = 0;

p2.length = 20;

options = optimset('GradObj','on','TolFun', minfunc,'Display','off');


%% Compute

for iter = 1:niter

    
a.max = 3;
a.min = 0;
x0 = Utils.rrand([2,1],a);      % Randomize initial point

tic
[x1,~,~,stats1] = rprop(funcgrad,x0,p);
t1(iter)=toc;

tic
[x2,~,~,stats2] = fminunc(funcgrad,x0,options);
t2(iter)=toc;

%tic
%[X, stats3, i] = minimize(x0, funcgrad,p2);
%t3(iter)=toc;

end


%% Plot results

fprintf('Average Running time to reach an Obj. value of %2.0e:\n',minfunc)
fprintf('Rprop: %f\n',mean(t1));
fprintf('Fminunc: %f\n',mean(t2));

figure()
Utils.rplot(@plot,{t1,t2})
legend(p.method,'fminunc')
xlabel('Experiment number')
ylabel('Time (sec)')

