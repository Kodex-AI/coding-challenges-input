import itertools


def anagram_naive(main_str, fetch_str):
    for perm in itertools.permutations(fetch_str):
        if ''.join(perm) in main_str:
            return True
    return False


def anagram_wrong(main_str, fetch_str):
    if len(fetch_str) == 1:
        result = fetch_str in main_str
        # print(result, fetch_str, main_str)
        return fetch_str in main_str
    fetch_list = list(fetch_str)
    for fetch_char in fetch_list:
        for i in range(len(main_str)):
            if fetch_char == main_str[i]:
                fetch_list_reduced = fetch_list.copy()
                fetch_list_reduced.remove(fetch_char)
                if (i - len(fetch_list_reduced) > 0 and anagram(main_str[i - len(fetch_list_reduced)-1:i], ''.join(fetch_list_reduced))) or (i + len(fetch_list_reduced) < len(main_str) and anagram(main_str[i:i+len(fetch_list_reduced)+1], ''.join(fetch_list_reduced))):
                    return True
    return False


def anagram_01(main_str, fetch_str):
    fetch_list = list(fetch_str)
    for fetch_char in fetch_list:
        for i in range(len(main_str)):
            if fetch_char == main_str[i]:
                fetch_list_reduced = fetch_list.copy()
                fetch_list_reduced.remove(fetch_char)
                if build_string(main_str[i:], fetch_list_reduced) or build_string(main_str[:i][::-1], fetch_list_reduced):
                    return True
    return False


def build_string(main_str, fetch_list):
    if len(fetch_list) == 0:
        return True
    if len(main_str) == 0:
        return False
    if len(main_str) == 1:
        return len(fetch_list) == 1 and main_str[0] == fetch_list[0]
    if main_str[0] in fetch_list:
        fetch_list_reduced = fetch_list.copy()
        fetch_list_reduced.remove(main_str[0])
        return build_string(main_str[1:], fetch_list_reduced)
    return False


def anagram(main_str, fetch_str):
    n = len(fetch_str)
    for i in range(len(main_str) - n):
        sub_string = main_str[i:i+n]
        if sorted(sub_string) == sorted(fetch_str):
            return True
    return False


def test():
    print(anagram('ihavelivedthebestlife', 'bees'))
    print(anagram('ihavelivedthebestlife', 'hide'))
    print(anagram('ihavelivedthebestlife', 'devil'))
    print(anagram('ihavelivedthebestlife', '012345678901'))
    print(anagram('ihavelivedthebestlife', 'bvdeilstvieehalte'))


if __name__ == '__main__':
    test()
    # print(list('dddd'))