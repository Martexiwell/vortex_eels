%  Eigenmodes for a sphere
clc
clear all;
close all;
%%

%number of triangular elements discretizing the sphere
npts = 300;
 
%  options for BEM simulation; nev is number of eigenmodes (eigenvalues)
op = bemoptions('sim','stat','nev', 20);

%  table of dielectric functions
epstab = { epsconst( 1 ), epsdrude( 'Au' ) };
%epstab = { epsconst( 1 ), epsconst( 1 ) };

%  Sphere parameters
diameter = 30;  %  diameter

% Defining the sphererical particle + its meshing
p = trisphere( npts, diameter ) ;

%  set up COMPARTICLE objects
p = comparticle( epstab, { p }, [ 2, 1 ], 1, op );

enei = 527;
%% plot the particle + its meshing 
%figure(1)
%plot( p, 'EdgeColor', 'b' );

%%  BEM simulation
%  initialize BEM solver; we use eigensolver
bemeig = bemstateig( p, op );

% from the eigensolution, we can get the corresponding pseudocharges on the
% surface corresponding to different eigenmodes
eigencharges = bemeig.ur;

for nEigenmode = 2:4
                %desired mode index   
sig = compstruct(p,enei);         %compstruct object that can be fed to the "potential" function
sig.sig = bemeig.ur(:,nEigenmode);    %replacement of an ordinary charge by the pseudocharge

%%% plot of the potential using the compoint object
% x = linspace( - diameter, diameter, 101 );
% z = linspace( - diameter, diameter, 101 );
% [X, Z] = ndgrid( x, z );
% ysel = 0; % coordinate of the selected y plane
% Y = ones( size(X) )*ysel;
% 
% pt = compoint( p, [ X(:), Y(:), Z(:) ], 'medium', 1, 'mindist', 1e-6 );      
% g = greenfunction( pt, p, op );
% pot = potential( g, sig );
% 
% phiOut = zeros(size(X));
% tmp = struct(pt);
% SubIndex = tmp.ind{1};
% phiOut(SubIndex) = pot.phi1;
% 
% pt = compoint( p, [ X(:), Y(:), Z(:) ], 'medium', 2, 'mindist', 1e-6 );      
% g = greenfunction( pt, p, op );
% pot = potential( g, sig );
% 
% phiIn = zeros(size(X));
% tmp = struct(pt);
% SubIndex = tmp.ind{2};
% phiIn(SubIndex) = pot.phi1;
% 
% phiInOut = phiIn + phiOut;
% 
% figure(2);
% hold off;
% imagesc( x, z, phiInOut ); 
% colorbar;  colormap jet;
% xlabel( 'x (nm)' );
% ylabel( 'z (nm)' );
% set( gca, 'YDir', 'norm' );
% axis equal tight

%% extracting potential in 3D
x = linspace( - diameter, diameter, 101 );
y = linspace( - diameter, diameter, 101 );
z = linspace( - diameter, diameter, 101 );
[X, Y, Z] = ndgrid( x, y, z );

pt = compoint( p, [ X(:), Y(:), Z(:) ], 'medium', 1, 'mindist', 1e-6 );      
g = greenfunction( pt, p, op );
pot = potential( g, sig );

phiOut = zeros(size(X));
tmp = struct(pt);
SubIndex = tmp.ind{1};
phiOut(SubIndex) = pot.phi1;

pt = compoint( p, [ X(:), Y(:), Z(:) ], 'medium', 2, 'mindist', 1e-6 );      
g = greenfunction( pt, p, op );
pot = potential( g, sig );

phiIn = zeros(size(X));
tmp = struct(pt);
SubIndex = tmp.ind{2};
phiIn(SubIndex) = pot.phi1;

phiInOut = phiIn + phiOut;

B=reshape(phiInOut,101,101*101);
fname = sprintf('Documents/MATLAB/MNPBEM17/MY_PROJECTS/OUTPUT/potential_sphere_eig%d.dat', nEigenmode);
%open(fname, 'w')
save(fname,'B','-ascii')

figure(nEigenmode)
plot(p, sig.sig)
end