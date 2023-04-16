clear all; clc; close all;
% Planteamiento de Función Mult Objetivo
% x: peso total del vehículo (en kg)
% y: potencia del motor (en kW)
f1 = @(x,y) 10.*(x)./y; % Consumo de combustible (en litros por cada 100 
                        % km recorridos)
f2 = @(x,y) 0.05.*x.^2+5.*(y); %Costo de producción (en miles de dólares)
f3 = @(x,y) 60+0.1.*x+0.2.*y.^2; %Nivel de ruido (en decibelios)
% Construir un coche que sea barato, comodo y rentable
MultiObj.fun=@(x)[f1(x(:,1),x(:,2)), f2(x(:,1),x(:,2)), f3(x(:,1),x(:,2))];
MultiObj.nVar = 2;
%El peso total del vehículo debería estar en el rango de 500 a 2000 kg,
%mientras que la potencia del motor debería estar en el rango de 50 a 200kW 
MultiObj.var_min = [500, 50];
MultiObj.var_max = [2000, 200];

% Parametros
params.Np = 200;        % Tamaño de la población
params.pc = 0.9;        % Probabilidad de cruce
params.pm = 1/6;        % Probabilidad de mutación: 1/3N, 
                        % donde N es el número de variables
params.maxgen = 10;    % Numero de generaciones
params.ms = 0.05;       %  fuerza de la mutación


%LLamo la función NSGA II, guardo los resultados
[fit, rank] =NSGA2(params,MultiObj);
%------------------------------------------------------------------
%La función fue implementada por:
%   Author:  Victor Martinez Cagigal                                      
%   Date:    22/12/2017                                                   
%   E-mail:  vicmarcag (at) gmail (dot) com 
%   Se le agrego una grafica donde se ve la ultima generación de la
%   frontera de pareto
% ----------------------------------------------------------------------- 
%   Referencias:                                                           
%    [1] Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002)
%        A fast and elitist multiobjective genetic algorithm: NSGA-II.    
%        IEEE transactions on evolutionary computation, 6(2), 182-197.    
%    [2] Víctor Martínez-Cagigal (2023). Non Sorting Genetic Algorithm II 
%        (NSGA-II) (https://www.mathworks.com/matlabcentral/fileexchange/
%        65494-non-sorting-genetic-algorithm-ii-nsga-ii), MATLAB Central 
%        File Exchange. Retrieved April 16, 2023.
