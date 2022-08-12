# Evaluación de fidelidad

Dados los estilos $S$, llamamos a la sucesión de *rolls* del estilo $s\in S$ como 
$$
\{r^s\}_i = r^s_1, r^s_2, ..., r^s_{N_s}
$$
con $N_s$ como la cantidad de *rolls* de dicho estilo.



Definimos la transferencia de estilo $T_{m,D,t}(r^o)=r'^{ot}$ como
$$
T_{m,D,t} = DEC(EMB_{m,D}(r^o) + AVG_{r\in \{R^t\}_i}\{EMB_{m,D}(r)\} - AVG_{r\in \{R^o\}_i}\{EMB_{m,D}(r)\})
$$
con $o,t\in S$ estilos original y objetivo respectivamente, $m$ modelo entrenado sobre el *dataset* $D$.



A partir de esto, calculamos el ránking de cercanía (en cuanto a plagio) del *roll* original vs. el transformado y el resto de los *rolls* del estilo. Es decir, definimos
$$
rank(i,o,t) = RANK(r'^{ot}_i, \{|r^o_i - x| : x\in R_i\})
$$
con $R_i = \{R^o\}_j - \{r_i^o\} \cup\{r_i'^{ot}\}$ (los *rolls* del estilo $o$ sin contar a $r_i$ junto con el $r_i$ transformado al estilo $t$).



Finalmente, calculamos la fidelidad de los *rolls* $\{r^{o}\}_i$ del estilo $o$ al transformarlos a $t$ con el modelo $m$ entrenado con el *dataset* $D$ como:
$$
fidelity(m,D,o,t) = \{1-\frac{rank(i,o,t)}{N_s} / i=1..N_s\}
$$
