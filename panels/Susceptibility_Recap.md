# Susceptibility

$$
\mathcal{S} = \int (-\lambda) \phi^q - \lambda \vec{\phi}^T \sigma_1 \vec{\phi} - i \text{Tr} \ln \left[i(G_0^{-1} + 2\lambda \hat{\phi} -\hat{u})\right] = \newline
= \int dt ~ (-\lambda) \phi^q - \lambda \vec{\phi}^T \sigma_1 \vec{\phi} - i\text{Tr} \ln \left[iG_0^{-1} \right] -i \text{Tr} \ln \left[ I - G_0(-2\lambda \hat{\phi} +\hat{u})\right] \newline
\approx \int dt ~  \delta \phi^T  (-\lambda \sigma_1) \delta\phi + \delta \phi^T (-4\lambda^2 \chi_0) \delta \phi + \delta \phi^T(2\lambda \chi_0)u + u^T (2\lambda \chi_0) \delta \phi - u^T \chi_0 u
$$

$$
\mathcal{S} \approx - u^T \chi_0 u - \delta \phi^T  (\lambda \sigma_1 +4\lambda^2 \chi_0) \delta \phi + \delta \phi^T(2\lambda \chi_0)u + u^T (2\lambda \chi_0) \delta \phi \newline
= - u^T \chi_0 u + u^T 4\lambda^2 \chi_0 [\lambda \sigma_1 + 4\lambda^2 \chi_0]^{-1} \chi_0 u = - u^T \left( \chi_0 - 4\lambda^2 \chi_0 [\lambda \sigma_1 + 4\lambda^2 \chi_0]^{-1} \chi_0 \right) u \newline
= - u^T(\chi_0 - \chi_0 U_{RPA} \chi_0)u
$$

$$
U_{RPA} = 4\lambda^2[\lambda \sigma_1 + 4\lambda^2 \chi_0]^{-1} = \left[ (4\lambda)^{-1}\sigma + \chi_0 \right]^{-1} = \newline
= \left( I + (4\lambda)\sigma_1 \chi_0\right)^{-1} (4\lambda) \sigma_1 \newline
\equiv (I + \alpha \sigma_1 \chi_0)^{-1} \alpha \sigma_1
$$

$$
\chi_{RPA} = \chi_0 ( I - U_{RPA} \chi_0) = \chi_0 (I - (I + \alpha\sigma_1 \chi_0)^{-1} \alpha \sigma_1 \chi_0) = \chi_0(I + \alpha \sigma_1 \chi_0)^{-1} \newline
\equiv \chi_0 (I + (4\lambda) \sigma_1 \chi_0)^{-1}
$$

$$
(I + \alpha \sigma_1 \chi_0)^{-1} = 
\begin{pmatrix}
I + \alpha \chi^R_0 & \alpha \chi^K_0 \\
0 & I + \alpha \chi^A_0
\end{pmatrix}^{-1} \newline
=  \begin{pmatrix}
(1 + \alpha \chi^R_0)^{-1} & (1 + \alpha \chi^R_0)^{-1}\alpha\chi^K_0 (1 + \alpha \chi^A_0)^{-1}  \\
0 &  (1 + \alpha \chi^A_0)^{-1} 
\end{pmatrix}
$$

$$
\chi_{RPA} = 
\begin{pmatrix}
0 & \chi^{A} \\
\chi^R & \chi^K
\end{pmatrix}_{RPA} =
\begin{pmatrix}
0 & \chi^{A} \\
\chi^R & \chi^K
\end{pmatrix}_0 .
 \begin{pmatrix}
(1 + \alpha \chi^R_0)^{-1} & (1 + \alpha \chi^R_0)^{-1}\alpha\chi^K_0 (1 + \alpha \chi^A_0)^{-1}  \\
0 &  (1 + \alpha \chi^A_0)^{-1} 
\end{pmatrix} \newline
= 
\begin{pmatrix}
0 & \chi^A_0 (I + (4\lambda) \chi^A_0)^{-1} \\
\chi^R_0 (I + (4\lambda) \chi^R_0)^{-1} & (I + (4\lambda) \chi^A_0)^{-1} \chi^K_0 (I + (4\lambda) \chi^R_0)^{-1}
\end{pmatrix}
$$

