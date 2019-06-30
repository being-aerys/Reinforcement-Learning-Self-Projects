from tensorboardX import SummaryWriter
import math

if __name__ == "__main__":

    writer = SummaryWriter() #create a writer

    #Lets visualize something

    for angle in range(-180, 180):
        angle_in_radian = angle * math.pi / 180
        value_of_sin_function = math.sin(angle_in_radian)

        #add the scalar to the write
        writer.add_scalar("sin function", value_of_sin_function, angle_in_radian)

    writer.close()#----------------------------------------------------------------dont forget to close the writer


    #Lets do multiple of these
    #create a list of functions to visualize
    writer_new = SummaryWriter()
    funcs_list = {"sin":math.sin, "cos":math.cos, "tan":math.tan}

    for angle in range(-180, 180):
        angle_in_radian = angle * math.pi / 180
        for function_name,function in funcs_list.items():

            value_of_function = function(angle_in_radian)

            # add the scalar to the write
            writer.add_scalar(function_name, value_of_sin_function, angle_in_radian)

    writer_new.close()  # ----------------------------------------------------------------dont forget to close the writer




