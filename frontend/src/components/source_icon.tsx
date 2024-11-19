import React from 'react'

import confluence from 'images/confluence.png'
import docs from 'images/docs.png'
import dropbox from 'images/dropbox.png'
import excel from 'images/excel.png'
import onedrive from 'images/onedrive.png'
import pdf from 'images/pdf.png'
import github from 'images/github.png'
import sharepoint from 'images/sharepoint.png'
import sheets from 'images/sheets.png'
import slides from 'images/slides.png'
import teams from 'images/teams.png'
import sql_server from 'images/sql server.png'
import word from 'images/word.png'
import faq from 'images/faq.png'
import xlsx from 'images/xlsx.png'
import docx from 'images/docx.png'
import pptx from 'images/pptx.png'
import defaultIcon from 'images/default.png'

export type SourceIconType = {
  className?: string
  icon:
    | 'confluence'
    | 'docs'
    | 'dropbox'
    | 'excel'
    | 'onedrive'
    | 'pdf'
    | 'sharepoint'
    | 'sheets'
    | 'slides'
    | 'teams'
    | 'sql_server'
    | 'word'
    | 'github'
    | 'faq'
    | 'docx'
    | 'pptx'
    | 'xlsx'
    | string
}
export const SourceIcon: React.FC<SourceIconType> = ({ className, icon }) => {
  const iconNameToImageMap = {
    confluence,
    docs,
    dropbox,
    excel,
    onedrive,
    pdf,
    sharepoint,
    sheets,
    slides,
    teams,
    sql_server,
    word,
    github,
    faq,
    docx,
    pptx,
    xlsx,
    default: defaultIcon,
  }

  // Use the default icon if the mapped icon does not exist
  const iconSrc = iconNameToImageMap[icon] || iconNameToImageMap.default

  return (
    <span className={className}>
      <img className="w-6 h-6" src={iconSrc} alt={icon} />
    </span>
  )
}
