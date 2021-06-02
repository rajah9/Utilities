from ParserUtil import ParserUtil
import logging
import sys
sys.path.insert(0, "../..")

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Sas(ParserUtil):
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

    def p_proc_end(self, p):
        '''
        procend : RUN EOL
        '''
        logger.debug('encountered RUN statement')

if __name__ == '__main__':
    s = Sas()
    s.run()