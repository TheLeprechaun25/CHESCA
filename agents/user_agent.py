from checa.agent import Checa

###################################################################
#####                Specify your agent here                  #####
###################################################################


class my_agent(Checa):
    """ Can be any subclass of citylearn.agents.base.Agent  """
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)


    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        return self.predict(observations, )

    def predict(self, observations, deterministic=True):
        """ Just a passthrough, can implement any custom logic as needed """
        return super().predict(observations, deterministic=deterministic)


###################################################################
SubmissionAgent = my_agent
###################################################################

