from .engine_ext import Parameter, State, Constant, ScopeSpec, Module, EquationSpec, allow_implicit



class SimpleCycle(Module):
    """Class implementing a consumer in a general way. This class can be extended to create more specific consumer types.
        Note that if that process heat can be defined in this class. This is practically done mostly with KB consumer
        """
    class SimpleCycleVariables(ScopeSpec):
        power_heat = Parameter(0)
        set = Parameter(0)
        COP = Parameter(4)
        eta_thermal = Parameter(0.93)
        T_cond_in = Parameter(40)
        T_evap_in = Parameter(9)
        F_cond = Parameter(10)
        F_evap = Parameter(10)
        R_cond = Parameter(1e-5)
        R_evap = Parameter(1e-5)
        T_super = Parameter(5)
        T_cond = State(30)
        T_evap = State(0)
        T_cond_mid = State(30)
        T_evap_mid = State(0)

        def generate_variables(self, equation: EquationBase):
            [add_power_with_energy(equation, p) for p in ['P_elec_compressor',
                                                      'P_heat_compressor',
                                                      'P_elec_heat_added', 'P_heat', 'P_cool']]

            [equation.add_parameter(k, 20) for k in [
                                                     'T_evap_out', 'T_cond_out',
                                                     'T_evap_exp', 'T_cond_exp']]

    tag = "cycle"

    default = SimpleCycleVariables()
    @staticmethod
    @EquationSpec(default)
    def diff_heat_pump(scope):

        #Use compressor function to calculate compressor performance at the required relative load
        scope.T_evap = scope.T_evap_mid - scope.T_super

        #scope.P_heat_compressor = scope.power_heat * scope.set
        #scope.P_elec_compressor = scope.P_heat_compressor/scope.COP

        # Calculate the power at the condenser
        scope.P_elec_heat_added = scope.P_elec_compressor * scope.eta_thermal
        scope.P_heat = scope.P_heat_compressor + scope.P_elec_heat_added
        # Cooling power at evaporator
        scope.P_cool = -scope.P_heat_compressor


        scope.T_cond_dot, scope.T_cond_out, scope.T_cond_exp = iterate_internal_temp(scope.T_cond_in, scope.T_cond,
                                                                                     scope.P_heat, scope.R_cond,
                                                                                     scope.F_cond)

        scope.T_evap_mid_dot, scope.T_evap_out, scope.T_evap_exp = iterate_internal_temp(scope.T_evap_in, scope.T_evap_mid,
                                                                                     scope.P_cool, scope.R_evap,
                                                                                     scope.F_evap)

    def attach_compressor(self, T_evap,
                          T_cond, P_heat_compressor, P_elec_compressor):
        # scope.m_dot_compressor, scope.P_heat_compressor, scope.P_elec_compressor = self.compressor(scope.T_evap,
        #                                                                                           scope.T_cond, scope.set)

        T_evap += self.default.T_evap_in
        T_cond += self.default.T_cond_in

        self.default.P_elec_compressor += P_elec_compressor
        self.default.P_heat_compressor += P_heat_compressor
