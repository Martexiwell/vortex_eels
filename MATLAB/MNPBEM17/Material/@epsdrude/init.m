function obj = init( obj )
%  Initialize Drude dielectric function.

%  atomic units
hartree = 27.2116;              %  2 * Rydberg in eV
tunit = 0.66 / hartree;         %  time unit in fs

switch obj.name
  case { 'Au', 'gold' }
    rs = 3;                     %  electron gas parameter in bohr radii
    obj.eps0 = 10;              %  background dielectric constant
    gammad = tunit / 10;        %  Drude relaxation rate
  case { 'Ag', 'silver' }
    rs = 3;
    obj.eps0 = 3.3;
    gammad = tunit / 30;
  case { 'Al', 'aluminum' }
    rs = 2.07;
    obj.eps0 = 1;
    gammad = 1.06 / hartree;
  case { 'myAg', 'my_silver' }      %omega_p = 9.1 * const.eV/const.hbar
                                    %gamma_p = 0.15 * const.eV/const.hbar
    rs = ((3 * ( 1.05457182e-34)^2 / (4 * pi * 8.8541878128e-12 * 9.1093837e-31 * 9.1^2) )^(1/3) ) / 5.29177210903e-11 ;
    obj.eps0 = 1;
    gammad = 0.15 / hartree;
  otherwise
    error( 'Material name unknown' );r
end

%  density in atomic units
density = 3 / ( 4 * pi * rs ^ 3 );
%  plasmon energy
wp = sqrt( 4 * pi * density );

%  save values
obj.gammad = gammad * hartree;
obj.wp     = wp     * hartree;

    
