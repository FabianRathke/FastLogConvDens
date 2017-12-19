function [N M] = getMN(dim,sparse)
if sparse 
    switch(dim)
    case 1
        N = 10; M = 5;
    case 2
        N = 12; M = 4;
    case 3
        N = 6; M = 4;
    case 4
        N = 4; M = 3;
    case 5
        N = 5; M = 2;
    case 6
        N = 4; M = 2;
    case 7
        N = 3; M = 2;
    case 8
        N = 2; M = 2;
    case 9
        N = 2; M = 2;
    otherwise
        error('Dimension of sample is to high. Please reduce dimensionality to at most 9 dimensions.');
    end
else
   switch(dim)
    case 1
        N = 10; M = 50;
    case 2
        N = 10; M = 10;
    case 3
        N = 9; M = 5;
    case 4
        N = 6; M = 4;
    case 5
        N = 5; M = 3;
    case 6
        N = 4; M = 3;
    case 7
        N = 3; M = 3;
    case 8
        N = 3; M = 2;
    case 9
        N = 2; M = 2;
    otherwise
        error('Dimension of sample is to high. Please reduce dimensionality to at most 9 dimensions.');
    end
end


