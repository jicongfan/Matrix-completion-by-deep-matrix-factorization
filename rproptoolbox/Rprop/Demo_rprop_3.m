% DEMO_RPROP_3 Rprop with GPU acceleration
%   Show the use of GPU acceleration for the Rprop function and compare its
%   performances with the normal CPU-computed version
%

%   Copyright (c) 2011 Roberto Calandra
%   $Revision: 0.60 $


%% Init

numdim = [1000000 500000 100000 50000 10000 5000 1000];

p.verbosity     = 1;                % Increase verbosity to print something
p.MaxIter       = 300;              % Maximum number of iterations
p.d_Obj         = 10e-12;           % Desired objective value


%% Compute

t = 1;

for i = numdim
    
    a.max = 3;
    a.min = 0;
    x0 = Utils.rrand([i,1],a);
    
    % with GPU
    funcgrad = @costfunction_gpu;       % Function to optimize
    p.useGPU = true;                    % use GPU acceleration if possible?
    p.funcgradgpu = true;               % does the cost function accept and
                                        % return variables as gpuArray?
    [x1,~,~,stats1] = rprop(funcgrad,x0,p);   
    
    
    % with CPU
    funcgrad = @costfunction;           % Function to optimize
    p.useGPU = false;                   % use GPU acceleration if possible?
    [x2,~,~,stats2] = rprop(funcgrad,x0,p);
    
    
    res.time1(t) = stats1.time(end);
    res.time2(t) = stats2.time(end);
    
    t = t+1;
    
end


%% Plot results

figure()
Utils.rplot(@loglog,{numdim,numdim},...
    {res.time1, res.time2})
legend('GPU','CPU','Location','SouthEast')
ylabel('Time (s)')
xlabel('Number of  parameters')

