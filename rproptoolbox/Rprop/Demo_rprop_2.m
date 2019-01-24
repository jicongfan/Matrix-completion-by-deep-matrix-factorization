% DEMO_RPROP_2 Compare the four Rprop methods
%
%

%   Copyright (c) 2011 Roberto Calandra
%   $Revision: 0.55 $


%% Init

funcgrad = @costfunction;       % Function to optimize

a.max = 3;
a.min = 0;
x0 = Utils.rrand([5000,1],a);

p.verbosity = 1;                    % Increase verbosity to print something
p.MaxIter   = 100;                  % Maximum number of iterations
p.display   = 0;


%% Compute

p.method = 'Rprop-';            % Define algorithm to use
[x1,~,~,stats1] = rprop(funcgrad,x0,p);

p.method = 'Rprop+';            % Define algorithm to use
[x2,~,~,stats2] = rprop(funcgrad,x0,p);

p.method = 'IRprop-';           % Define algorithm to use
[x3,~,~,stats3] = rprop(funcgrad,x0,p);

p.method = 'IRprop+';           % Define algorithm to use
[x4,~,~,stats4] = rprop(funcgrad,x0,p);


%% Plot results

figure()
Utils.rplot(@semilogy,{stats1.error, stats2.error, stats3.error, stats4.error})
legend('Rprop-','Rprop+','IRprop-','IRprop+','Location','SouthWest')
xlabel('Number of iterations')
ylabel('Obj. Value')

figure()
Utils.rplot(@semilogy,{stats1.time, stats2.time, stats3.time, stats4.time},...
    {stats1.error, stats2.error, stats3.error, stats4.error})
legend('Rprop-','Rprop+','IRprop-','IRprop+','Location','SouthWest')
xlabel('Time (s)')
ylabel('Obj. Value')

drawnow

