function [ind, max_vol, it] = maxvol_spsd(A, n, r, do_update, debug)
% Compute the indices ind of the  r x r principal submatrix in the n x n SPSD matrix A that maximizes det(A(ind, ind))
%
%---------------------------------------INPUT--------------------------------------------------------------------------------------------------------
%
% A1		target matrix in full format or as handle function  
% n			size of A
% r 		size of the sought dominant submatrix
% do_update (optional) applies the updates based on the Woodbury identity for b-1 iterations in a row (b = 1 --> disable the updates, b = inf --> always update)
% debug		(optional) allows some debugging print
% 
%---------------------------------------OUTPUT--------------------------------------------------------------------------------------------------------
%
% ind		indices of the principal submatrix
% max_vol	volume associated with the selected indices
% it 		numer of iterations performed
%
%-----------------------------------------------------------------------------------------------------------------------------------------------------
	tol = 5e-2;
	maxit = 50;
	if ~exist('debug', 'var')
		debug = 0;
	end
	if ~exist('do_update', 'var')
		do_update = 1;
	end
	ind = []; max_vol = 0;
	% As starting point we choose the outcome of r steps of ACA 
	[~, ind] = aca_spsd(A, n, 0, r);
	ind = sort(ind); 

	changed = true;
	it = 1;
	if debug
		fprintf('It: 0, vol = %e, true vol = %e, cond(A) = %1.2e\n', max_vol, det(A(ind, ind)), cond(A(ind, ind)))
	end
	while it <= maxit
		X = zeros(n - r, r);
		if it == 1
			cind = setdiff([1:n], ind);
			R = chol(A(ind, ind)); 
		end
		if mod(it, do_update) == mod(1, do_update)
			B = (A(:, ind) / R) / R';
			iA = diag(R \ (R' \ eye(r)));
		end

		c = zeros(n - r, 1); 
		for j = 1:n - r
			c(j) = B(cind(j), :) * A(ind, cind(j));
		end

		% Vectorized computation of the determinants
		if isfloat(A)
			dA = diag(A); dA = c - dA(cind);
		else
			dA = zeros(n - r, 1);
			for i = 1:n - r
				dA(i) = c(i) - A(cind(i), cind(i));
			end
		end
		
		X = (dA * iA.' - abs(B(cind, :)).^2);

		[mx0, big_ind] = max(abs(X(:))); % retrieve maximum element in X
		[i0, j0] = ind2sub([n - r, r], big_ind);
		if  mx0 <= 1 + tol % if the volume does not increase enough, then we return
			changed = false;
    	else
			changed = true;
			ej = zeros(r, 1); ej(j0) = 1;

			% Apart from the update of the Cholesky factor, we apply the updates based on the Woodbury identity 		
			% accordinf to do_update

			if mod(it, do_update) ~= 0
				% update of D 
				vBu = [(ej' / R) / R'; B(cind(i0), :) - B(ind(j0), :)]; 
				vBv = [iA(j0), B(cind(i0), j0); B(cind(i0), j0), c(i0) - A(cind(i0), cind(i0))] \ vBu;
				iA = iA - diag(vBu' * vBv);
			end

			% update of the Cholesky factor
			R = updateR(R, [ej, A(ind, cind(i0)) - A(ind, ind(j0))], [A(ind(j0), ind(j0)) + A(cind(i0), cind(i0)) - 2 * A(ind(j0), cind(i0)), 1; 1, 0]);
	
			if mod(it, do_update) ~= 0
				% update of B
				v = A(:, cind(i0)) - A(:, ind(j0));
				DB = v * ((ej' / R) / R') - (A(:, ind) * vBu') * vBv;
				B = B + DB;
			end
	
			% update the index set and the volume
			temp = ind(j0);
			ind(j0) = cind(i0); 
			cind(i0) = temp;
			max_vol = abs(max_vol * mx0);
		end 
		if debug
			fprintf('It: %d, vol = %e, true vol = %e, cond(A) = %1.2e\n', it, max_vol, det(A(ind, ind)), cond(A(ind, ind)))
		end
		if ~changed
			break
		end
		it = it + 1;
	end
	if it > maxit
		it = maxit;
		warning('MAXVOL_SPSD:: reached maximum number of iterations, gain = %f', mx0)
	end
end
%---------------------------------------------------------------------------------
function R = updateR(R, U, W)
% Compute the Cholesky factorization of R' * R + U * W * U' updating R
	[Z, D] = eig(W);
	U = U * Z;
	if D(1, 1) > 0
			R = cholupdate(R, U(:, 1) * sqrt(D(1, 1)));
			R = cholupdate(R, U(:, 2) * sqrt(-D(2, 2)), '-');
	else
			R = cholupdate(R, U(:, 2) * sqrt(D(2, 2)));
			R = cholupdate(R, U(:, 1) * sqrt(-D(1, 1)), '-');
		end
end
