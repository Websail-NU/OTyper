
def add_path(path, string, pattern):
    if pattern is None and path[0] != '*':
        return add_path_h(path, 0, string, dict())
    elif pattern is None and path[0] == '*':
        return add_path_h(path, 0, string, list())
    else:
        return add_path_h(path, 0, string, pattern)

def add_path_h(path, pos, string, pattern):
    if path[pos] != '*':
        if pos == len(path) - 1:
            if not path[pos] in pattern:
                pattern[path[pos]] = string
            return pattern
        else:
            if not path[pos] in pattern and path[pos+1] != '*':
                pattern[path[pos]] = dict()
            elif not path[pos] in pattern and path[pos+1] == '*':
                pattern[path[pos]] = list()
            pattern[path[pos]] = add_path_h(path, pos+1, string, pattern[path[pos]])

            return pattern
    else:
        assert(pattern is None or type(pattern) is list)
        if pos == len(path) - 1:
            if not string in pattern:
                pattern.append(string)
            return pattern
        else:
            if pattern is None or len(pattern) == 0:
                if path[pos+1] != '*':
                    ret = add_path_h(path, pos+1, string, dict())
                    pattern.append(ret)
                else:
                    ret = add_path_h(path, pos+1, string, list())
                    pattern.append(ret)
            else:
                ret = add_path_h(path, pos+1, string, pattern[0])
                pattern[0] = ret
            return pattern


def unit_test():
    pattern = add_path(['a'], '1', None)
    pattern = add_path(['b'], '2', pattern)
    print pattern

    pattern = add_path(['a','*'], '1', None)
    pattern = add_path(['b','*'], '2', pattern)
    print pattern

    pattern = add_path(['*'], '1', None)
    pattern = add_path(['*'], '2', pattern)
    print pattern

    pattern = add_path(['a','*','b'], '1', None)
    pattern = add_path(['a','*','c'], '2', pattern)
    print pattern


if __name__ == "__main__":
    unit_test()
