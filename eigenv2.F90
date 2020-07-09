#include "fintrf.h"

subroutine mexFunction(nlhs, plhs, nrhs, prhs)
	implicit none

	integer :: nlhs, nrhs, j, jmin, jmax, h, hmin, hmax, i
	mwPointer :: plhs(*), prhs(*), mxCreateDoubleMatrix
	double precision, allocatable :: D(:,:), u(:)
	double precision :: c,s, v(2)
	character(len=256) :: buffer
	mwPointer :: n, mxGetN, mxGetPr

	n = mxGetN(prhs(1))

	allocate(D(n,n), u(n))

	call mxCopyPtrToReal8(mxGetPr(prhs(1)), D, n*n)
	call mxCopyPtrToReal8(mxGetPr(prhs(2)), u, n)

	do j = n,2,-1
		![G, u(j - 1:j)] = planerot(u(j - 1:j));
		call drotg(u(j-1), u(j), c, s)
        jmin = max(1, j-2); jmax = min(n, j+1);
		!D(j - 1:j, jmin:jmax) = G * D(j - 1:j, jmin:jmax);
		call drot(jmax - jmin + 1, D(j-1, jmin), n, D(j, jmin), n, c, s)
		!D(jmin:jmax, j - 1:j, :) = D(jmin:jmax, j - 1:j) * G';
		call drot(jmax - jmin + 1, D(jmin, j-1), 1, D(jmin, j), 1, c, s)
		!call printm(D, n)
		do h = j + 1,n ! Bulge chasing
			hmin = max(1, h-2); hmax = min(n, h+1);
			![G, ~] = planerot(D(h - 1 :h, h - 2));
			v = D(h-1:h, h-2);
			call drotg(v(1), v(2), c, s)
			!D(h - 1:h, hmin:hmax) = G * D(h - 1:h, hmin:hmax);
			call drot(hmax - hmin + 1, D(h-1, hmin), n, D(h, hmin), n, c, s)
			!D(hmin:hmax, h - 1:h) = D(hmin:hmax, h - 1:h) * G';
			call drot(hmax - hmin + 1, D(hmin, h-1), 1, D(hmin, h), 1, c, s)
			!u(h - 1:h) = G * u(h - 1:h);
			call drot(1, u(h-1), 1, u(h), 1, c, s)
		end do
	end do
	D(1, 1) = D(1, 1) - u(1) **2;
	!D = triu(tril(D, 1), -1);




	plhs(1) = mxCreateDoubleMatrix(n, n, 0)
	call mxCopyReal8ToPtr(D, mxGetPr(plhs(1)), n*n)

	deallocate(D, u)
end
subroutine printm(A, n)
	integer :: n, i
	double precision :: A(n, n)
	character(len=256) :: buffer
	do i = 1,n
		write(buffer, *) A(i, :), '\n'
		call mexPrintf(buffer)
	end do
	write(buffer, *) '\n'
	call mexPrintf(buffer)
end
