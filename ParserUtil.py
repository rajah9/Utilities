import logging
import sys
sys.path.insert(0, "../..")

import ply.lex as lex
import ply.yacc as yacc
import os

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ParserUtil:
    """
    Base class for a lexer/parser that has the rules defined as methods.
    """
    tokens = ()
    keywords = ()
    precedence = ()

    def __init__(self, **kw):
        self.debug = kw.get('debug', 0)
        self.names = []
        try:
            modname = os.path.split(os.path.splitext(__file__)[0])[
                          1] + "_" + self.__class__.__name__
        except:
            modname = "parser" + "_" + self.__class__.__name__
        self.debugfile = modname + ".dbg"

        # build the lexer and parser
        lex.lex(module=self, debug=self.debug)
        yacc.yacc(module=self, debug=self.debug, debugfile=self.debugfile)

    def run(self):
        while True:
            try:
                s = input('parse > ')
            except EOFError:
                break
            if not s:
                continue
            yacc.parse(s)

class HelloWorld(ParserUtil):
    # All keywords must be uppercase
    # Helpful to have the following tuple end in an extra comma.
    keywords = (
        'HELLO',
    )
    # Helpful to have the following tuple end in an extra comma.
    tokens = keywords + (
        'NAME',
        'COMMA',
    )

    # Tokens
    t_HELLO = '(?i)hello'
    t_COMMA = ','

    def t_NAME(self, t):
        r'[A-Za-z][A-Za-z]*'
        tupper = t.value.upper()
        if tupper in self.keywords:
            t.type = tupper
            logger.debug(f'interpreting as keyword: {t.value}')
        else:
            logger.debug(f'encountering name: {t.value}')
        return t

    t_ignore = " \t"

    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += t.value.count('\n')

    def t_error(self, t):
        logger.error(f'Illegal character {t.value[0]}')
        t.lexer.skip(1)

    # Parser rules
    def p_statement_greet(self, p):
        '''
        statement : HELLO
                  | HELLO NAME
                  | HELLO COMMA NAME
        '''
        lexdata = " "
        for lexeme in p:
            if lexeme:
                lexdata += str(lexeme) + " "
        logger.debug(f'greeting encountered: {lexdata}')

    def p_error(self, p):
        if p:
            logger.error(f'Syntax error at {p.value}')
        else:
            logger.error('Syntax error at EOF')

if __name__ == '__main__':
    p = HelloWorld()
    p.run()
