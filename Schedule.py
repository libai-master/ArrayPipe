def Generate_scheduling():
    Scheduling_list=[[]]
    with open("/root/Arraypipe/ModelParallel/util_lib/schedule.txt","r+") as sequuence:
        for _ in sequuence:
            GPU_Sch=sequuence.read()
            Scheduling_list.append(GPU_Sch)
    return Scheduling_list

#Gpipe
# Scheduling_list=[[[0,'f'],[1,'f'],[2,'f'],[3,'f'],[4,'f'],[5,'f'],[6,'f'],[6,'b'],[5,'b'],[4,'b'],[3,'b'],[2,'b'],[1,'b'],[0,'b']],
#                      [[0,'f'],[1,'f'],[2,'f'],[3,'f'],[4,'f'],[5,'f'],[6,'f'],[6,'b'],[5,'b'],[4,'b'],[3,'b'],[2,'b'],[1,'b'],[0,'b']],
#                      [[0,'f'],[1,'f'],[2,'f'],[3,'f'],[4,'f'],[5,'f'],[6,'f'],[6,'b'],[5,'b'],[4,'b'],[3,'b'],[2,'b'],[1,'b'],[0,'b']],
#                      [[0,'f'],[1,'f'],[2,'f'],[3,'f'],[4,'f'],[5,'f'],[6,'f'],[6,'b'],[5,'b'],[4,'b'],[3,'b'],[2,'b'],[1,'b'],[0,'b']]]
#     return Scheduling_list