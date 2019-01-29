###############################
Introduction to AITP
###############################

.. _lets-keep-it-simple:

***********************
Lets Keep It Simple
***********************

A training platform for AI **models**

Code snipet
=======

::

    pragma solidity >=0.4.0 <0.6.0;

    contract SimpleStorage {
        uint storedData;

        function set(uint x) public {
            storedData = x;
        }

        function get() public view returns (uint) {
            return storedData;
        }
    }


.. note::
    All identifiers (contract names, function names and variable names) are restricted to
    the ASCII character set. It is possible to store UTF-8 encoded data in string variables.

.. warning::
    Be careful with using Unicode text, as similar looking (or even identical) characters can
    have different code points and as such will be encoded as a different byte array.

.. index:: ! examples

Examples
===================

code reference:  ``address public minter;`` declares a state variable of type address

.. index:: mapping  

The next line, ``mapping (address => uint) public balances;`` also

.. index:: event

The line ``event Sent(address from, address to, uint amount);`` 


.. _second-subsection:

*****************
Second Subsection
*****************

WIP
