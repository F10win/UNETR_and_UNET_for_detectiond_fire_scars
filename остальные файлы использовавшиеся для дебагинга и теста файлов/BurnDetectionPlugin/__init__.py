# -*- coding: utf-8 -*-
"""
/***************************************************************************
 BurnDetectionPlugin
                                 A QGIS plugin
 Detect forest burns using UNETR
                             -------------------
        begin                : 2026-05-13
        copyright            : (C) 2026 by Your Name
        email                : your.email@example.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

def classFactory(iface):
    """Load BurnDetectionPlugin class"""
    from .burn_detection_plugin import BurnDetectionPlugin
    return BurnDetectionPlugin(iface)