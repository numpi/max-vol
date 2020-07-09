function [ind, max_vol, it] = maxvol_ratio(A1, A2, n, r, do_update, debug)
% Compute the indices ind of the  r x r principal submatrices in the n x n SPSD matrices A1 and A2 that maximizes det(A1(ind, ind)) / det(A2(ind, ind))
%
%---------------------------------------INPUT--------------------------------------------------------------------------------------------------------
%
% A1,A2		target matrices in full format or as handle function  
% n			size of A1 and A2
% r 		size of the sought dominant submatrix
% do_update (optional) applies the updates based on the Woodbury identity for b-1 iterations in a row (b = 1 --> disable the updates, b = inf --> always update)
% debug		(optional) allows some debugging print
% 
%---------------------------------------OUTPUT--------------------------------------------------------------------------------------------------------
%
% ind		indices of the principal submatrices
% max_vol	ratio of volumes associated with the selected indices
% it 		numer of iterations performed
%
%-----------------------------------------------------------------------------------------------------------------------------------------------------
	tol = 5e-2;
	maxit = 100;
	if ~exist('debug', 'var')
		debug = 0;
	end
	if ~exist('do_update', 'var')
		do_update = 1;
	end
	ind = []; max_vol = 0;
	% As starting point we choose the outcome of r steps of ACA_RATIO 
	[ind, max_vol] = aca_ratio(A1, A2, n, r, 0, 0, 1);
	ind = sort(ind); 

	changed = true;
	it = 1;
	if debug
		fprintf('It: 0, vol = %e, true vol = %e, cond(A1) = %1.2e, cond(A2) = %1.2e\n', max_vol, det(A1(ind, ind))/det(A2(ind, ind)), cond(A1(ind, ind)), cond(A2(ind, ind)))
	end
	while it <= maxit
		X = zeros(n - r, r);
		if it == 1
			cind = setdiff([1:n], ind);
			R1 = chol(A1(ind, ind)); R2 = chol(A2(ind, ind)); 
		end
		if mod(it, do_update) == mod(1, do_update)
			B1 = (A1(1:n, ind) / R1) / R1';
			B2 = (A2(1:n, ind) / R2) / R2';
			iA1 = diag(R1 \ (R1' \ eye(r)));
			iA2 = diag(R2 \ (R2' \ eye(r)));
		end

		c1 = zeros(n - r, 1); c2 = c1; 
		for j = 1:n - r
			c1(j) = B1(cind(j), :) * A1(ind, cind(j));
			c2(j) = B2(cind(j), :) * A2(ind, cind(j));
		end

		% Vectorized computation of the determinant ratios
		if isfloat(A1)
			dA1 = diag(A1); dA1 = c1 - dA1(cind);
		else
			dA1 = zeros(n - r, 1);
			for i = 1:n - r
				dA1(i) = c1(i) - A1(cind(i), cind(i));
			end
		end
		if isfloat(A2)
			dA2 = diag(A2); dA2 = c2 - dA2(cind);
		else
			dA2 = zeros(n - r, 1);
			for i = 1:n - r
				dA2(i) = c2(i) - A2(cind(i), cind(i));
			end
		end
		X = (dA1 * iA1.' - abs(B1(cind, :)).^2) ./ (dA2 * iA2.' - abs(B2(cind, :)).^2);

		[mx0, big_ind] = max(abs(X(:))); % retrieve maximum element in X
		[i0, j0] = ind2sub([n - r, r], big_ind);
		if  mx0 <= 1 + tol % if the ratio of volumes does not increase enough, then we return
			changed = false;
    	else
			changed = true;
			ej = zeros(r, 1); ej(j0) = 1;


			% Apart from the update of the Cholesky factor, we apply the updates based on the Woodbury identity 		
			% accordinf to do_update
			
			if mod(it, do_update) ~= 0
				% update of D 
				vB1u = [(ej' / R1) / R1'; B1(cind(i0), :) - B1(ind(j0), :)]; 
				vB1v = [iA1(j0), B1(cind(i0), j0); B1(cind(i0), j0), c1(i0) - A1(cind(i0), cind(i0))] \ vB1u;
				iA1 = iA1 - diag(vB1u' * vB1v);
				vB2u = [(ej' / R2) / R2'; B2(cind(i0), :) - B2(ind(j0), :)];
				vB2v = [iA2(j0), B2(cind(i0), j0); B2(cind(i0), j0), c2(i0) - A2(cind(i0), cind(i0))] \ vB2u;
				iA2 = iA2 - diag(vB2u' * vB2v);
			end

			% update of the Cholesky factor
			R1 = updateR(R1, [ej, A1(ind, cind(i0)) - A1(ind, ind(j0))], [A1(ind(j0), ind(j0)) + A1(cind(i0), cind(i0)) - 2 * A1(ind(j0), cind(i0)), 1; 1, 0]);
			R2 = updateR(R2, [ej, A2(ind, cind(i0)) - A2(ind, ind(j0))], [A2(ind(j0), ind(j0)) + A2(cind(i0), cind(i0)) - 2 * A2(ind(j0), cind(i0)), 1; 1, 0]);
	
			if mod(it, do_update) ~= 0
				% update of B
				v1 = A1(1:n, cind(i0)) - A1(1:n, ind(j0));
				DB1 = v1 * ((ej' / R1) / R1') - (A1(1:n, ind) * vB1u') * vB1v;
				B1 = B1 + DB1;
				v2 = A2(1:n, cind(i0)) - A2(1:n, ind(j0));
				DB2 = v2 * ((ej' / R2) / R2') - (A2(1:n, ind) * vB2u') * vB2v;
				B2 = B2 + DB2;
			end
	
			% update the index set and the volume
			temp = ind(j0);
			ind(j0) = cind(i0); 
			cind(i0) = temp;
			max_vol = abs(max_vol * mx0);
		end 
		if debug
			fprintf('It: %d, vol = %e, true vol = %e, cond(A1) = %1.2e, cond(A2) = %1.2e\n', it, max_vol, det(A1(ind, ind))/det(A2(ind, ind)), cond(A1(ind, ind)), cond(A2(ind, ind)))
		end
		if ~changed
			break
		end
		it = it + 1;
	end
	if it > maxit
		it = maxit;
		warning('MAXVOL_RATIO:: reached maximum number of iterations, gain = %f', mx0)
	end
end
%---------------------------------------------------------------------------------
function R = updateR(R, U, W)
% Compute the Cholesky factorization of R' * R + U * W * U' updating R
	[Z, D] = eig(full(W));
	if issparse(R)
		R = full(R);
	end
	U = U * Z;
	if D(1, 1) > 0
			R = cholupdate(R, U(:, 1) * sqrt(D(1, 1)));
			R = cholupdate(R, U(:, 2) * sqrt(-D(2, 2)), '-');
	else
			R = cholupdate(R, U(:, 2) * sqrt(D(2, 2)));
			R = cholupdate(R, U(:, 1) * sqrt(-D(1, 1)), '-');
		end
end
