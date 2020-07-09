function traces = arnoldi_trace_powers_update(A, U, k, debug)
% Return the approximation of trace((A - U(:, i) * U(:, i)')^j - A^j), j=1...k, i = 1,...,t computed with the Lanczos method 
% run simultaneously on the columns of U
%
%---------------INPUT----------------------------------------------------------------------------------------------------------------
%
% A               Hermitian matrix argument   
% U               low-rank factorization of the updates
% k				  maximum power to be computed
% debug           (optional) allows to print either the heuristic or the true error (the second one is expensive), default debug = 0  
%
%---------------OUTPUT--------------------------------------------------------------------------------------------------------------
%
% traces          (t x k)-array containing trace((A - U(:, i) * U(:, i)')^j - A^j), j = 1,...,k  i = 1,...,t
%
%-----------------------------------------------------------------------------------------------------------------------------------
	if ~exist('debug', 'var')
		debug = 0;
	end
	t = size(U, 2);
	traces = zeros(t, k);

	if k <= 4 % hardcoding of the cases k = 1,2,3,4 
		UU = [U, zeros(size(U, 1), (k - 1) * size(U, 2))];
		UAU = zeros(t, k);
		for j = 2:k
			UU(:, (j - 1) * t + 1: j * t) = A * UU(:, (j - 2) * t + 1: (j - 1) * t);
		end
		for j = k:-1:1
			UAU(:, j) = diag(UU(:, 1:t)' * UU(:, (j - 1) * t + 1: j * t));
		end
		traces(:, 1:k) = (-UAU(:, 1)) .^ [1:k];
		if k == 1
			return
		end
		for j = 2:k
			traces(:, j) = traces(:, j) + j * sum( (UAU(:, 1) .^ [j - 2:-1:0]) .* UAU(:, 2:j) .* ((-1).^ [j-1:-1:1]), 2);
		end
		if k == 4
			traces(:, 4) = traces(:, 4) + 2 * UAU(:, 2).^2;
		end
		return
	end

     
    for j = 1:k+1
		% Augment the Krylov space
		if j == 1 %~exist('Um', 'var')
			[Um, HA, param_A, lucky] = poly_krylov_sim(A,  U); 
			Cm = -vecnorm(U).^2; 
		else
			[Um, HA, param_A, lucky] = poly_krylov_sim(Um, HA, param_A);
		end
		
		if lucky 
			warning('ARNOLDI_TRACE_POWERS_UPDATE:: Detected lucky breakdown')
			break
		end
	end 

	% Retrieve the projection of A into the  Krylov subspace
	for i = 1:t
		Gm = HA{i}(1:end - 1, :);
		tGm = Gm; tGm(1, 1) = tGm(1, 1) + Cm(i);
		% Compute the trace of the core factor of the update
		d1 = sort(eig(tGm));
		d2 = sort(eig(Gm));
		traces(i, :) =  sum(d1.^[1:k] - d2.^[1:k], 1);
	
		if debug	
			true_traces = zeros(k, 1);
			for j = 1:k
				true_traces(j) = trace((A - U(:, i) * U(:, i)')^j - A^j);
				fprintf('A^%d: err = %e\n', j, abs(true_traces(j) - traces(i, j))); 
			end
			pause
		end
	end
end






