def get_lr_decay_parameters(parameters, learning_rate, groups: dict):
    custom_lr_parameters = dict((group_name, {'params': [], 'lr': learning_rate * lr_factor})
                                for (group_name, lr_factor) in groups.items())
    custom_lr_parameters['default'] = {'params': [], 'lr': learning_rate}

    for parameter_name, parameter in parameters:
        matches = False
        for group_name, lr in groups.items():
            if str.startswith(parameter_name, group_name):
                custom_lr_parameters[group_name]['params'].append(parameter)
                matches = True
                break

        if not matches:
            custom_lr_parameters['default']['params'].append(parameter)

    return custom_lr_parameters.values()
