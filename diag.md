Traceback (most recent call last):
  File "<frozen runpy></frozen>", line 198, in _run_module_as_main
  File "<frozen runpy></frozen>", line 88, in _run_code
  File "/home/alist/opt/code_aster_17.4.0/lib/python3.11/site-packages/mpi4py/__main__.py", line 7, in <module></module>
    main()
  File "/home/alist/opt/code_aster_17.4.0/lib/python3.11/site-packages/mpi4py/run.py", line 230, in main
    run_command_line(args)
  File "/home/alist/opt/code_aster_17.4.0/lib/python3.11/site-packages/mpi4py/run.py", line 47, in run_command_line
    run_path(sys.argv[0], run_name='__main__')
  File "<frozen runpy></frozen>", line 291, in run_path
  File "<frozen runpy></frozen>", line 98, in _run_module_code
  File "<frozen runpy></frozen>", line 88, in _run_code
  File "/mnt/c/Users/alist/source/repos/Lamb/Gairezi/Aster/cases/whole_model_crack.comm", line 70, in <module></module>
    RESU = STAT_NON_LINE(
           ^^^^^^^^^^^^^^
  File "/home/alist/opt/code_aster_17.4.0/lib64/aster/code_aster/Supervis/ExecuteCommand.py", line 183, in run
    return cmd.run_(**kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/home/alist/opt/code_aster_17.4.0/lib64/aster/code_aster/Supervis/ExecuteCommand.py", line 226, in run_
    self.exec_(keywords)
  File "/home/alist/opt/code_aster_17.4.0/lib64/aster/code_aster/Supervis/ExecuteCommand.py", line 516, in exec_
    self._call_oper(syntax)
  File "/home/alist/opt/code_aster_17.4.0/lib64/aster/code_aster/Supervis/ExecuteCommand.py", line 282, in _call_oper
    return libaster.call_oper(syntax, 0)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
libaster.ConvergenceError:
 ╔════════════════════════════════════════════════════════════════════════════════════════════════╗
 ║ <EXCEPTION></exception> <MECANONLINE9_7>                                                                   ║
 ║                                                                                                ║
 ║ Arrêt pour cause d'absence de convergence avec le nombre d'itérations requis dans l'algorithme ║
 ║ non-linéaire de Newton.                                                                        ║
 ║    La base globale est sauvegardée. Elle contient les pas archivés avant l'arrêt.              ║
 ║                                                                                                ║
 ║    Conseils :                                                                                  ║
 ║    - Augmentez ITER_GLOB_MAXI.                                                                 ║
 ║    - Réactualisez plus souvent la matrice tangente.                                            ║
 ║    - Raffinez votre discrétisation temporelle.                                                 ║
 ║    - Essayez d'activer la gestion des événements (découpe du pas de temps par exemple) dans la ║
 ║ commande DEFI_LIST_INST.                                                                       ║
 ║    - Vérifier la cohérence des paramètres entre code_aster et MFRONT si vous utilisez une loi  ║
 ║ MFRONT.                                                                                        ║
 ╚════════════════════════════════════════════════════════════════════════════════════════════════╝
