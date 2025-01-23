# Parallel processing with Karabo

Karabo streamlines the process of setting up an environment for parallelization. Through its utility function `parallelize_with_dask`, Karabo nudges the user towards a seamless parallelization experience. By adhering to its format, users find themselves in a `pit of success` with parallel processing. This ensures efficient task distribution across multiple cores or even entire cluster nodes, especially when handling large datasets or tasks with high computational demands.

## Points to Consider When Using `parallelize_with_dask` and Dask in General

When leveraging the `parallelize_with_dask` function for parallel processing in Karabo, users should keep in mind the following best practices:

1. **Avoid Infinite Tasks**: Ensure that the tasks you're parallelizing have a defined end. Infinite or extremely long-running tasks can clog the parallelization pipeline.

2. **Beware of Massive Tasks**: Large tasks can monopolize resources, potentially causing an imbalance in the workload distribution. It's often more efficient to break massive tasks into smaller, more manageable chunks.

3. **No Open h5 Connections**: Objects with open h5 connections are not `pickleable`. This means that they cannot be serialized and sent to other processes. If you need to pass an object with an open h5 connection to a function, close the connection before passing it to the function, e.g. by calling `h5file.close()` or `.compute()` inside Karabo.

4. **Use `.compute()` on Dask Arrays**: Before passing Dask arrays to the function, call `.compute()` on them to realize their values. This avoids potential issues and ensures efficient processing.

5. **Refer to Dask's Best Practices**: For a more comprehensive understanding and to avoid common pitfalls, consult [Dask's official best practices guide](https://docs.dask.org/en/stable/best-practices.html).

Following these guidelines will help ensure that you get the most out of Karabo's parallel processing capabilities.


## Parameters
- iterate_function (callable): The function to be applied to each element of the iterable. This function should take the current element of the iterable as its first argument, followed by any specified positional and keyword arguments.

- iterable (iterable): The collection of elements over which the iterate_function will be applied.

- args (tuple): Positional arguments that will be passed to the iterate_function after the current element of the iterable.

- kwargs (dict): Keyword arguments that will be passed to the iterate_function.

## Returns
- tuple: A tuple containing the results of the iterate_function for each element in the iterable. Results are gathered using Dask's compute function.

## Additional Notes
It's important when working on a `Slurm Cluster` to call DaskHandler.setup() at the beginning.

If 'verbose' is specified in kwargs and is set to True, progress messages will be printed during processing.

The function internally uses the distributed scheduler of Dask.

Leverage the `parallelize_with_dask` utility in Karabo to harness the power of parallel processing and speed up your data-intensive operations.

## Function Signature

```python
from karabo.util.dask import DaskHandler

# Example
def my_function(element, *args, **kwargs):
    # Do something with element
    return result

DaskHandler.parallelize_with_dask(my_function, my_iterable, *args, **kwargs) # The current element of the iterable is passed as the first argument to my_function
>>> (result1, result2, result3, ...)
```

## Use Karabo on a SLURM cluster

Karabo manages all available nodes through Dask, making the computational power conveniently accessible for the user. The `DaskHandler` class streamlines the creation of a Dask client and offers a user-friendly interface for interaction. This class contains static variables to modify the behavior of a Dask client, if they've changed before creating a client. 

While users are not required to interact with Dask directly - thanks to the background processes managed by Karabo - the Dask client should be initialized at the beginning of your script with `DaskHandler.setup` (see example below). This has to do with the spawning of new processes when creating `Nanny` processes.

```python
from karabo.util.dask import DaskHandler

if __name__ == "__main__":
    # Get the Dask client
    client = DaskHandler.get_dask_client() # Not needed anymore to call .setup()

    # Use the client as needed
    result = client.submit(my_function, *args)
```
```python
from karabo.util.dask import DaskHandler

if __name__ == "__main__":
    DaskHandler.setup()
    result = <function_of_karabo_which_uses_dask_in_the_background>(*args)
```

Disable the usage of Dask by Karabo.

```python
from karabo.util.dask import DaskHandler
# Modify the static variables
DaskHandler.use_dask = False
```

Please also check out the `DaskHandler` under `karabo.util.dask` for more information.

## Dask Dashboard
The Dask dashboard link should be printed in stdout. Just copy the link into your browser, and then you're able to observe the current dask-process. If you run Karabo on a VM without access to a browser and internet, you can use ssh `port forwarding` to access the Dask Dashboard from your local machine (e.g. `ssh -N -L <local-port>:(<remote-node>:)<remote-port> <host>`). Don't forget to use the `<local-port>` in the browser-link if you used port-forwarding. In `VSCODE`, this can be done directly when using the "PORTS" tab; just paste the IP address and port number from stdout into the "Port" column and click on "Open in Browser" in the "Local Address" column.
