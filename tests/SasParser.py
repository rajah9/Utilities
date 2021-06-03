from ParserUtil import ParserUtil
from CompilerUtil import SasCompilerUtil
import logging
import sys

_PROC_MEANS = 'MEANS'

sys.path.insert(0, "../..")

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SasParser(ParserUtil):
    _curr_proc_type = None

    def __init__(self):
        super(SasParser, self).__init__()
        self._scu = SasCompilerUtil()

    @property
    def curr_proc_type(self):
        return self._curr_proc_type

    @curr_proc_type.setter
    def curr_proc_type(self, proc_type: str):
        logger.debug(f'Starting PROC {proc_type}')
        self._curr_proc_type = proc_type

    # All keywords must be uppercase
    # Helpful to have the following tuple end in an extra comma.
    keywords = (
        'PROC',
        'MEANS',
        'DATA',
        'EOL',
        'RUN',
    )
    # Helpful to have the following tuple end in an extra comma.
    tokens = keywords + (
        'DATASETNAME',
        'EQUALS',
#        'ID',
    )

    # Tokens
    t_EOL = ';'
    t_EQUALS = '='

    def t_DATASETNAME(self, t):
        r'[A-Za-z][A-Za-z0-9\.]*'
        tupper = str(t.value).upper()
        if tupper in self.keywords:
            t.type = tupper
            logger.debug(f'interpreting as keyword: {t.value}')
        else:
            logger.debug(f'encountering datasetname: {t.value}')
        return t

    # Parsing rules
    def p_statement_proc(self, p):
        '''
        statement : procdecl
        '''

    def p_procdecl(self, p):
        '''
        procdecl : procmeans
        '''

    def p_procmeans(self, p):
        '''
        procmeans : procmeansdecl procend
        '''

    def p_proc_means_decl(self, p):
        '''
        procmeansdecl : PROC MEANS EOL
                      | PROC MEANS procoptions EOL
        '''
        logger.debug(f'About to write PROC MEANS with this dictionary: {self._scu.proc_options}')
        self._scu.init_options()

    def p_proc_options(self, p):
        '''
        procoptions : procoptions procoption
                    | procoption
        '''

    def p_proc_option(self, p):
        '''
        procoption : dataoption
        '''

    def p_data_option(self, p):
        '''
        dataoption : DATA EQUALS DATASETNAME
        '''
        for i in range(len(p)):
            logger.debug(f'encountering DATA option. Param {i}: {p[i]}')
        self._scu.add_option(option='DATA', value=p[3])

    def p_proc_end(self, p):
        '''
        procend : RUN EOL
        '''
        logger.debug('encountered RUN statement')

if __name__ == '__main__':
    s = SasParser()
    s.run()