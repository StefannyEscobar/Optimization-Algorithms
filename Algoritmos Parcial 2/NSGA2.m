% ----------------------------------------------------------------------- %
% La función NSGAII realiza un Algoritmo Genético-II de No Ordenamiento %
% para funciones continuas. %
% %
% Parámetros de entrada: %
% - params: Estructura que contiene los parámetros personalizados.%
% * params.Np: Número de cromosomas en la población. %
% * params.maxgen: Número máximo de generaciones. %
% * params.pc: Probabilidad de cruzamiento. %
% * params.pm: Probabilidad de mutación. %
% * params.ms: Fuerza de la mutación (alrededor del 2% %
% al 10% está bien). %
% - MultiObj: Estructura que contiene los parámetros relativos a las%
% funciones de optimización. %
% * MultiObj.fun: Función anónima de varios objetivos a %
% minimizar. %
% * MultiObj.nVar: Número de variables. %
% * MultiObj.var_min: Vector que indica los valores mínimos %
% del espacio de búsqueda en cada dimensión.%
% * MultiObj.var_max: Igual que 'var_min' con los máximos. %
% ----------------------------------------------------------------------- %
% Para un ejemplo de uso, ejecute 'ejemplo.m'. %
% ----------------------------------------------------------------------- %
% Autor: Victor Martinez-Cagigal %
% Fecha: 25/11/2019 %
% E-mail: vicmarcag (at) gmail (dot) com %
% Versión:1.2 %
% Log: %
% - 1.0: Versión inicial (21/12/2017). %
% - 1.1: El algoritmo Fast Non Sorting ahora está vectorizado %
% para mejorar el rendimiento (mucho menos tiempo de %
% cálculo) (22/12/2017). %
% - 1.2: El antiguo operador de mutación se sustituye por la %
% adición de una distribución normal ponderada, tal %
% como sugiere Alexander Hagg, que proporciona una mejor %
% convergencia (25/11/2019). %
% ----------------------------------------------------------------------- %
% Referencias: %
% [1] Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002)%
% A fast and elitist multiobjective genetic algorithm: NSGA-II. %
% IEEE transactions on evolutionary computation, 6(2), 182-197. %
% ----------------------------------------------------------------------- %
function [Rfit,Rrank] = NSGA2(params,MultiObj)
% Parámetros
    Np = params.Np; % Número de cromosomas en la población
    maxgen = params.maxgen; % Número máximo de generaciones
    pc = params.pc; % Probabilidad de cruzamiento
    pm = params.pm; % Probabilidad de mutación
    ms = params.ms; % Fuerza de la mutación
    fun = MultiObj.fun; % Función objetivo
    nVar = MultiObj.nVar; % Número de variables (dimensiones u objetivos)
    var_min = MultiObj.var_min(:);  % Valor minimo para cada generacion
    var_max = MultiObj.var_max(:);  % Valor maximo para cada generacion
        
    %% Initializacion
    gen   = 1; %La primera generacion en 1
    %Generacion de matriz aleatoria  tamaño (Np x nVar) donde cada fila 
    % representa una posible solución al problema.se hace tomando una muestra 
    % aleatoria uniforme de valores entre var_min y var_max para cada 
    % variable de decisión. 
    P     = repmat((var_max-var_min)',Np,1).*rand(Np,nVar) + repmat(var_min',Np,1);
    %Calcula la función de aptitud (fitness) 
    % de cada posible solución en la matriz P.
    Pfit  = fun(P);
    %Realiza una clasificación no dominada rápida, el nivel más bajo contiene
    % las soluciones más dominadas y el nivel más alto contiene las soluciones 
    % no dominadas, las soluciones 1er nivel son la frontera de pareto
    Prank = FastNonDominatedSorting_Vectorized(Pfit);
    %La cuarta línea selecciona los padres para la próxima generación
    %Elige las soluciones de los niveles mas altos, esto garantiza
    %diversidad genetica
    [P,~] = selectParentByRank(P,Prank);
    % Genera una nueva poblacion
    Q = applyCrossoverAndMutation(P,pc,pm,ms,var_max,var_min);
    % Graficos y mensajes
    % crea una figura 3D que muestra la ubicación de las soluciones no dominadas
    h_fig = figure(1);
    h_rep = plot3(Pfit(:,1),Pfit(:,2),Pfit(:,3),'ok'); hold on;
    grid on; xlabel('Consumo de combustible L/Km'); ylabel('Costo de producción $'); zlabel('Nivel de ruido dB');
    title('Optimizar construcción de un carro');
    drawnow;
    axis square;

    display(['Generación #' num2str(gen) ' - #Sol en primer frente: ' num2str(sum(Prank==1))]);

    %% Ciclo principal NSGA-II
    %Condicion de parada
    stopCondition = false;
    while ~stopCondition
        
        % Combina la poblacion actual P, con la generada Q paraa formar R
        R = [P; Q];
        
        % Se evalua la aptitud de la poblacion R y se calcula el rande de
        %Pareto igual que se hizo antes
        Rfit = real(fun(R));
        Rrank = FastNonDominatedSorting_Vectorized(Rfit);
        % Visualizamos el frente pareto actual
        figure(h_fig); delete(h_rep);
        h_rep = plot3(Rfit(1:Np,1),Rfit(1:Np,2),Rfit(1:Np,3),'ok'); hold on;

        grid on; xlabel('Consumo de combustible L/Km'); ylabel('Costo de producción $');
        drawnow;
        axis square;
       
        %Imprimo la generacion y el numero de soluciones del primer frente
        %de pareto
        display(['Generacion #' num2str(gen) ' #Sol en primer frente: ' num2str(sum(Rrank==1))]);
        
        
        % Ordena la poblacion R por rango de Pareto y por aptitud 
        [Rrank,idx] = sort(Rrank,'ascend');
        Rfit = Rfit(idx,:);
        R = R(idx,:);
        
        % Calcula la distancia de hacinamiento para la población combinada R
        [Rcrowd,Rrank,~,R] = crowdingDistances(Rrank,Rfit,R);
        
        % Selecciona la población para la próxima generación utilizando
        % la selección basada en rango y distancia
        P = selectParentByRankAndDistance(Rcrowd,Rrank,R);
        
        % Genera una nueva población Q a partir de la población seleccionada
        % P mediante operaciones de cruce y mutación. o sea el hijo
        Q = applyCrossoverAndMutation(P,pc,pm,ms,var_max,var_min);
        
        % Aunmenta la generacion y comprueba si ya se alcanzo el #maximo de
        % generaciones
        gen = gen + 1;
        if(gen>maxgen), stopCondition = true; end
    end
    %disp('Frente de Pareto:');
    %disp(P(:,1:2)); % muestra los primeros tres objetivos de cada
    % individuo en el frente de Pareto
    % Se grafica la ultima generacion
    valoresx = P(:,1);
    valoresy = P(:,2);    
    valores = MultiObj.fun([valoresx, valoresy]); 
    scatter3(valores(:,1), valores(:,2) ,valores(:,3),10, 'r', 'filled');
end
%%Defino la funcion
% esta se encarga de seleccionar nuevos padres basándose en el operador 
% de distancia de agrupamiento/hacinamiento (crowding distance) y el ranking.


function [newParent] = selectParentByRankAndDistance(Rcrowd,Rrank,R)
    
    % Inicializacion

    %obtiene la longitud de la mitad de Rcrowd 
    % (que se dice es el mismo tamaño que la población)
    N = length(Rcrowd)/2;
    %Numero de frentes de pareto
    Npf = length(unique(Rrank));
    %Se inicializa una matriz de N*R
    newParent = zeros(N,size(R,2));
    
    % Seleccionar los cromosomas que van a estar en la matriz de newParent
    pf = 1;
    numberOfSolutions = 0;
    while pf <= Npf
        % Si hay espacio se seleccionan todas las soluciones en el rango
        % actual y se agregan  a newParent. También se actualiza el número
        % de soluciones seleccionadas.
        if numberOfSolutions + sum(Rrank == pf) <= N
            newParent(numberOfSolutions+1:numberOfSolutions+sum(Rrank == pf),:) = R(Rrank == pf,:);
            numberOfSolutions = numberOfSolutions + sum(Rrank == pf);
        % Sino, se ordenan en orden descendente y se agregan las mas
        % grandes
        else
            rest = N - numberOfSolutions;
            lastPF = R(Rrank == pf,:);
            lastPFdist = Rcrowd(Rrank == pf);
            [~,idx] = sort(lastPFdist,'descend');
            lastPF = lastPF(idx,:);
            newParent(numberOfSolutions+1:numberOfSolutions+rest,:) = lastPF(1:rest,:);
            numberOfSolutions = numberOfSolutions + rest;
        end
        % SE actualiza pf para passar al otro frente
        pf = pf + 1;
    end
end
%% Funcion que  calcula las distancias de hacinamiento(crowding)
% de cada punto en el frente de Pareto.
function [sortCrowd,sortRank,sortFit,sortPop] = crowdingDistances(rank,fitness,pop)
    
% Initializacion
    sortPop = [];
    sortFit = [];
    sortRank = [];
    sortCrowd = [];
    %Encontramos el numero total de frentes de pareto
    Npf = length(unique(rank));
    %Recorremos cada frente
    for pf = 1:1:Npf
        %Encontramos el inidice actual
        index = find(rank==pf);
        % extraemos las funciones objetivo
        temp_fit = fitness(index,:);
        temp_rank = rank(index,:);
        temp_pop = pop(index,:);
        
        %Ordenamos los puntos
        [temp_fit,sort_idx] = sortrows(temp_fit,1);
        %Agregamos las funciones objetivo, el rango y la pobracion
        %ordenados a las variables de salida
        temp_rank = temp_rank(sort_idx);
        sortFit = [sortFit; temp_fit];
        sortRank = [sortRank; temp_rank];
        sortPop = [sortPop; temp_pop(sort_idx,:)];
        
        % Calculamos las distancias de hacinamiento(crowding)para cada 
        % punto en el frente de pareto actual, lo hacemos para cada funcion
        % objetivo y es la suma de la distancia entre los puntos adyacentes 
        % dividida por el rango total de esa función objetivo.
        temp_crowd = zeros(size(temp_rank));
        for m = 1:1:size(fitness,2)
            temp_max = max(temp_fit(:,m));
            temp_min = min(temp_fit(:,m));
            for l = 2:1:length(temp_crowd)-1
                temp_crowd(l) = temp_crowd(l) + (abs(temp_fit(l-1,m)-temp_fit(l+1,m)))./(temp_max-temp_min);
            end
        end
        %Aca establecemos quee los puntos extremos tienen una distancia
        % de Hacinamiento(crowding) infinita.
        temp_crowd(1) = Inf;
        temp_crowd(length(temp_crowd)) = Inf;
        sortCrowd = [sortCrowd; temp_crowd];
    end
end
%% Funcion que  se encarga de generar una población de hijos aplicando 
% operaciones de crossover y mutación sobre la población de padres.

function Q = applyCrossoverAndMutation(parent,pc,pm,ms,var_max,var_min)
    % Parametros
    N = size(parent,1);
    nVar = size(parent,2);
    
    %  Se inicializa la poblacion de hijos
    Q = parent;
    
    % se seleccionan los individuos de la población que van a ser 
    % sometidos a crossover

    %Generamos un vector del tamaño del numero de padres
    cross_idx = rand(N,1) < pc;
    cross_idx = find(cross_idx);
    %Para cada padre c, si es 1 se selecciona otro padre al azar
    for c = 1:1:length(cross_idx)
        selected = randi(N,1,1);
        while selected == c
            selected = randi(N,1,1);
        end
        %Apartir de 2 padres seleccionados se generan 2 hijos intercambiando
        % los valores de los cromosomas a partir del punto de corte.
        cut = randi(nVar,1,1);
        Q(c,:) = [parent(c,1:cut), parent(selected,cut+1:nVar)];
    end
    
    % Se genera una poblacion mutada con la distribucion Gaussiana
    %con media 0 y desv 1 se calcula una mutación multiplicando
    %ms(es una escala para multiplicar el rango de valores permitidos 
    % para cada variable) por la diferencia entre var_max y var_min
    mutatedPop = Q + ms.*repmat((var_max-var_min)',N,1).*randn(N,nVar);
    minVal = repmat(var_min',N,1);
    maxVal = repmat(var_max',N,1);
    %Esta población mutada se limita entre los valores máximos y 
    % mínimos permitidos por los parámetros.
    mutatedPop(mutatedPop<minVal) = minVal(mutatedPop<minVal);
    mutatedPop(mutatedPop>maxVal) = maxVal(mutatedPop>maxVal);
    
    % se seleccionan los valores de la población mutada mutadedPop con una 
    % probabilidad pm(esta es dada por el problema)
    % y se actualiza la población de hijos Q con estos valores mutados.
    mut_idx = rand(N,nVar) < pm;
    Q(mut_idx) = mutatedPop(mut_idx);
end
%% Funcion que realiza una selección de padres mediante un torneo binario. 
% Se le pasa como entrada la población inicial P y el rango Prank de cada 
% individuo de la población.

function [P1,P1rank]   = selectParentByRank(P, Prank)
    % Se generan los vectores de los indices left_idx y right_idx de tamaño N 
    N = length(Prank);  
    %Se generan con valores aleatorios entre 1 y N
    left_idx  = randi(N,N,1);
    right_idx = randi(N,N,1);
    %Se indica cuales participan en el torneo
    while sum(left_idx==right_idx)>0

        right_idx(left_idx==right_idx) = randi(N,sum(left_idx==right_idx),1);
    end
    
    % Se hace el torneo comprando los randos de los individuos
    % correspondientes a cada indice
    winners = zeros(N,1);
    %En este método, se seleccionan aleatoriamente dos individuos de la 
    % población y se compara su aptitud (fitness). El individuo con mejor
    % aptitud se selecciona como ganador del torneo.
    winners(Prank(left_idx)<=Prank(right_idx)) = left_idx(Prank(left_idx)<=Prank(right_idx));
    winners(Prank(right_idx)<Prank(left_idx)) = right_idx(Prank(right_idx)<Prank(left_idx));
    
    % Finalmente se seleccionan los padres a aprtir de los indices de los
    % ganadores del torneo y los rangos de los padres seleccionados
    P1 = P(winners,:);
    P1rank = Prank(winners,:);
end
%% implementa el algoritmo Fast Non Dominated Sorting, que se utiliza en el
% contexto de optimización multiobjetivo. 
% Nota del autor:
% fitnesses. Note: the code is not vectorized, its programming is just
% based on Deb2002.

%La función recibe como entrada una matriz de fitness, donde cada fila representa 
% un individuo y cada columna representa una función objetivo.
function [RANK] = FastNonDominatedSorting_Loop(fitness)
    % Initializacion
    %Numero de soluciones en la poblacion, donde fitness es la matriz de
    %aptitud de la poblacion
    Np = size(fitness,1);
    %Vector del mismo  tamaño de la poblacion
    N = zeros(Np,1);
    %S es una celda vacía que almacenará las soluciones dominadas por cada 
    % solución, 
    S{Np,1} = [];
    %PF es una celda vacía que almacenará las soluciones que
    % pertenecen a cada frente de Pareto 
    PF{Np,1} = [];
    %RANK es un vector de NaNs del mismo tamaño que la población que se 
    % utilizará para almacenar el rango de Pareto de cada solución.
    RANK = NaN(Np,1);
    
    % Algortimo principal:
    
    for p_idx = 1:1:Np
        p = fitness(p_idx,:);
        for q_idx = 1:1:Np
            q = fitness(q_idx,:);
            %Si P domina a Q 
            if dominates(p,q)
                % Se anade Q a las soluciones dominadas por P
                S{p_idx,1} = [S{p_idx,1}; q_idx];
              % De lo contrario se incrementa el contador de soluciones no
              %dominadas de P
            elseif dominates(q,p)
                N(p_idx) = N(p_idx) + 1;
            end
        end
        if N(p_idx) == 0
            RANK(p_idx) = 1;
            PF{1,1} = [PF{1,1}; p_idx];
        end
    end
    % Segunda parte del algoritmo:
    %Buscamos asignar el rango a los individuos
    i = 1;
    %Se repite hasta que no queden individuos
    while ~isempty(PF{i,1})
        Q = [];
        %PF es la frontera actual
        currPF = PF{i,1};
        for p_idx = 1:1:length(currPF)
            %Se actualiza el conjunto dominante S 
            Sp = S{currPF(p_idx),1};
            for q_idx = 1:1:length(Sp)
                %Y el numero de soluciones no dominantes
                N(Sp(q_idx)) = N(Sp(q_idx))-1;
                if(N(Sp(q_idx)) == 0)
                    %Almacenamos el rango asignado a cada individuo
                    RANK(Sp(q_idx)) = i + 1;
                    %Luego actualizamos el numero de soluciones dominaadas
                    Q = [Q; Sp(q_idx)];
                end
            end
        end
        i = i + 1;
        PF{i,1} = Q;
    end
end
%% Funcion vectorizada de Fast Non Dominated Sorting
%Esta implementación en particular es vectorizada, lo que significa que
% utiliza operaciones matriciales y de vectorización en lugar de iteraciones 
% de bucles para calcular las clasificaciones de Pareto. Esto acelera
% significativamente el tiempo de cómputo en comparación con la implementación
% tradicional del algoritmo.
function [RANK] = FastNonDominatedSorting_Vectorized(fitness)
    % Initializacion
    %Un proceso similar al anterior
    Np = size(fitness,1);
    %En este caso RANK en 0
    RANK = zeros(Np,1);
    current_vector = [1:1:Np]';
    current_pf = 1;
    %se calcula un conjunto de todas las posibles combinaciones de pares de 
    % vectores de aptitud en la matriz all_perm.
    all_perm = [repmat([1:1:Np]',Np',1), reshape(repmat([1:1:Np],Np,1),Np^2,1)];
    all_perm(all_perm(:,1)==all_perm(:,2),:) = [];
    
    % Se compara cada vector de aptitud en la matriz de fitness con todos
    % los demás vectores de aptitud para determinar si es dominado o no.
    while ~isempty(current_vector)
        
        % se realiza un ciclo mientras el vector de partículas
        % "current_vector" no esté vacío.
        if length(current_vector) == 1
            %si es solo una particula se le asigna el rango actual "current_pf"
            % a esa partícula y se rompe el ciclo while. 
            RANK(current_vector) = current_pf;
            break;
        end
        %Nota del autor"
        % Non-dominated particles
            % Note: nchoosek has an exponential grow in computation time, so
            % it's better to take all the combinations including repetitions using a
            % loops (quasi-linear grow) or repmats (linear grow)
            %all_perm = nchoosek(current_vector,2);   
            %all_perm = [all_perm; [all_perm(:,2) all_perm(:,1)]];     
 
        %Primero se evaluan todas las posibles combinaciones con la funcion
        %Domiante y se almacenan los indices de las particulas dominadas
        d = dominates(fitness(all_perm(:,1),:),fitness(all_perm(:,2),:));
        dominated_particles = unique(all_perm(d==1,2));
        % se comprueba si ya no hay más espacio para otras fronteras de Pareto
        %Si no hay mas se rompe el while
        if sum(~ismember(current_vector,dominated_particles)) == 0
            break;
        end
        % Se actualizan las clasificaciones de los no dominantes en RANK
        non_dom_idx = ~ismember(current_vector,dominated_particles);
        RANK(current_vector(non_dom_idx)) = current_pf;

        %Se eliminan todas las combinaciones de pares que involucren partículas dominadas.   
        all_perm(ismember(all_perm(:,1),current_vector(non_dom_idx)),:) = [];
        all_perm(ismember(all_perm(:,2),current_vector(non_dom_idx)),:) = [];
        current_vector(non_dom_idx) = [];
        %Se incrementa el contador
        current_pf = current_pf + 1;
    end
end
%% Funcion que si todos los valores de x son menores o iguales a los valores 
% correspondientes de y, y al menos un valor de x es estrictamente menor que 
% el valor correspondiente de y.
function d = dominates(x,y)
    d = (all(x<=y,2) & any(x<y,2));
end

    