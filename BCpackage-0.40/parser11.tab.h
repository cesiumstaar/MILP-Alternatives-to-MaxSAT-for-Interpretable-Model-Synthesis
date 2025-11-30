/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_BCP11_PARSER11_TAB_H_INCLUDED
# define YY_BCP11_PARSER11_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int bcp11_debug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    LPAREN = 258,                  /* LPAREN  */
    RPAREN = 259,                  /* RPAREN  */
    LBRACKET = 260,                /* LBRACKET  */
    RBRACKET = 261,                /* RBRACKET  */
    SEMICOLON = 262,               /* SEMICOLON  */
    COMMA = 263,                   /* COMMA  */
    DEF = 264,                     /* DEF  */
    ASSIGN = 265,                  /* ASSIGN  */
    EQUIVf = 266,                  /* EQUIVf  */
    IMPLYf = 267,                  /* IMPLYf  */
    ORf = 268,                     /* ORf  */
    ANDf = 269,                    /* ANDf  */
    ODDf = 270,                    /* ODDf  */
    EVENf = 271,                   /* EVENf  */
    NOTf = 272,                    /* NOTf  */
    ITEf = 273,                    /* ITEf  */
    IMPLY = 274,                   /* IMPLY  */
    EQUIV = 275,                   /* EQUIV  */
    OR = 276,                      /* OR  */
    ODD = 277,                     /* ODD  */
    AND = 278,                     /* AND  */
    NOT = 279,                     /* NOT  */
    ID = 280,                      /* ID  */
    TRUE = 281,                    /* TRUE  */
    FALSE = 282,                   /* FALSE  */
    NUM = 283                      /* NUM  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 37 "parser11.y"

  char *charptr;
  int intval;
  Gate *gate;
  std::list<Gate*>* list;

#line 99 "parser11.tab.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE bcp11_lval;


int bcp11_parse (void);


#endif /* !YY_BCP11_PARSER11_TAB_H_INCLUDED  */
