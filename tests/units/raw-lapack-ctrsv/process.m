L = [1+2*i,0,0,0;2+4*i,3+6*i,0,0;4+8*i,5+10*i,6+12*i,0;7+14*i,8+16*i,9+18*i,10+20*i];
b = [3+7*i;6+4*i;9+7*i;10+5*i];
y = forward_substitution(L, b)
norm(L * y - b, 'fro') / norm(b, 'fro')
