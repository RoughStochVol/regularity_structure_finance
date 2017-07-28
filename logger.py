
def logger(simulation_id, parameter_dict):
    """
    Creates a .txt log file and writes the contents of a given dictionary of
    parameters (parameter_dict) into it.

    Parameters
    ==========

    :param simulation_id: simulation identifyer number
    :param parameter_dict: dictionary of parameter values

    :type simulation_id: int
    :type parameter_dict: dictionary
    """
    with open("log_%s.txt" % simulation_id, "w") as log_file:

        log_file.write("# A regularity structure for rough volatility. \n")
        log_file.write("# Simulation ID: %s \n\n" % simulation_id)

        for key, value in parameter_dict.items():

            log_file.write(key + ": " + str(value) + '\n')
