import argparse
import csv


class _Writer:
    def __init__(self, add_bench):
        self.__add_bench = add_bench

    def __addinfo(self, bench, data):
        if self.__add_bench:
            return {'Benchmark': bench, **data}
        return data

    def benchdata(self, bench, data):
        self._benchdata(self.__addinfo(bench, data))

    def finish(self):
        self._finish()


class _CSVWriter(_Writer):
    def __init__(self, f, data: dict, fieldnames=None):
        add_bench = fieldnames is None or 'Benchmark' in fieldnames and 'Benchmark' not in data
        super().__init__(add_bench)

        if fieldnames is None:
            fieldnames = ['Benchmark', *data['Total'].keys()]

        self.__csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        self.__csv_writer.writeheader()

    def _benchdata(self, data):
        self.__csv_writer.writerow(data)

    def _finish(self):
        pass


class _HumanWriter(_Writer):
    def __init__(self, f, data: dict, fieldnames=None):
        add_bench = fieldnames is None or 'Benchmark' in fieldnames and 'Benchmark' not in data
        super().__init__(add_bench)

        if fieldnames is None:
            fieldnames = ['Benchmark', *data['Total'].keys()]

        self.__f = f
        self.__fieldnames = fieldnames
        self.__data = {name: [f'{name}:'] for name in fieldnames}
        self.__num_entries = 1

    def _benchdata(self, data):
        self.__num_entries += 1
        for k, v in data.items():
            self.__data[k].append(str(v))

    def _finish(self):
        col_max = [max(len(self.__data[field][index]) for field in self.__fieldnames)
                   for index in range(self.__num_entries)]
        for field in self.__fieldnames:
            for index, val in enumerate(self.__data[field]):
                self.__f.write(f'{val:{col_max[index]+1}}')
            self.__f.write('\n')


def _write_data(writer: _Writer, data: dict):
    for bench, bench_data in data.items():
        writer.benchdata(bench, bench_data)
    writer.finish()


def write_csv(f, data: dict, *, fieldnames=None):
    _write_data(_CSVWriter(f, data, fieldnames), data)


def write_human(f, data: dict, *, fieldnames=None):
    _write_data(_HumanWriter(f, data, fieldnames), data)


def add_output_format_arg(parser: argparse.ArgumentParser, default='csv'):
    parser.add_argument('--format', default=default, choices=('csv', 'human'),
                        help=f'Which format style to use (default: {default})')
    FORMAT_OPTIONS = {
        'csv': write_csv,
        'human': write_human,
    }

    if not hasattr(parser, '__analyze_post_process_parse_args__'):
        setattr(parser, '__analyze_post_process_parse_args__', {})
    getattr(parser, '__analyze_post_process_parse_args__')['format'] = FORMAT_OPTIONS.__getitem__
