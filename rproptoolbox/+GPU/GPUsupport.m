% GPUSUPPORT Check if exist a GPU supported for computations
%   [SUPPORT] = GPUsupport() return a boolean that indicate whenever a GPU
%   that can be used for computations has been found or not. Whenever
%   multiple GPU can be used it automatically select the best one.
%

%   Copyright (c) 2012 Roberto Calandra
%   $Revision: 0.10 $


function [support] = GPUsupport(verbose)

if nargin<1
    verbose           = 0;            % [0-3] Verbose mode
end


%% Identify Devices
try
    
    % Number of GPU
    ngpu = gpuDeviceCount;
    if verbose
        fprintf ('Number of GPU(s): %d\n',ngpu)
    end
    
    % NVIDIA driver installed
    driverver = parallel.internal.gpu.CUDADriverVersion;
    if verbose>2
        fprintf ('NVIDIA driver version: %s\n',driverver)
    end
    
catch
    
    % No GPU?
    if verbose
        warning('Impossible to Identify GPU(s)')
    end
    support = false;
    return
    
end


%% Analyze GPU(s)

gpucapable = zeros([ngpu 1]);
for ii = 1:ngpu
    try
        m(ii) = gpuDevice(ii);
     
        gpucapable(ii)=m(ii).DeviceSupported;
        
        if verbose>1
            if gpucapable(ii)
                
                fprintf('GPU %d: %s with CUDA support (v.%s)\n',...
                    ii,m(ii).Name,m(ii).ComputeCapability)
                
            else
                
                fprintf('GPU %d: %s does NOT have CUDA support >1.3 (v.%s)\n',...
                    ii,m(ii).Name,m(ii).ComputeCapability)
                
            end
        end
        
    catch
        warning(['GPU ' num2str(ii) ' doesn"t respond'])
        
    end
end


%% Is there a GPU supported?

ngpusupported = sum(gpucapable);

if ngpusupported
    support = true;
    if verbose
        fprintf ('Supported GPU found\n',ngpu)
    end
else
    support = false;
    
    if verbose
        warning('No supported GPU found')
    end
end


%% Select best GPU for computations

if ngpusupported>1
    % based either on Gflops or Memory (and support)
    if verbose>1
        %fprintf('Selected GPU %i')
    end
    
end


end


